import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import weibull_min


def cvar(values, alpha=.95):
    values = np.sort(np.asarray(values))[::-1]
    mass = (1 - alpha) * len(values)
    n = int(mass)
    return float((values[:n].sum() + (mass - n) * values[min(n, len(values) - 1)]) / mass)


def prepare(c, out):
    rng = np.random.default_rng(c['seed'])
    jobs = []
    for j in range(c['jobs']):
        release = float(rng.uniform(0, c['shifts'] * c['shift_hours'] * .8))
        duration = rng.uniform(.5, 3.5, c['machines'])
        eligible = rng.random(c['machines']) > .25
        eligible[int(rng.integers(c['machines']))] = True
        jobs.append(dict(id=j, family=int(rng.integers(c['families'])), release=release,
                         due=release + float(rng.uniform(8, 40)), duration=duration.tolist(), eligible=eligible.tolist(),
                         weight=float(rng.uniform(1, 3))))
    (out / 'instance.json').write_text(json.dumps(dict(jobs=jobs)))
    true_shape = rng.uniform(1.5, 2.5, c['machines'])
    true_scale = rng.uniform(80, 160, c['machines'])
    samples = np.stack([weibull_min.rvs(k, scale=s, size=c['lifetime_samples'], random_state=rng) for k, s in zip(true_shape, true_scale)])
    np.savez_compressed(out / 'lifetimes.npz', lifetime=samples, event=np.ones_like(samples), true_shape=true_shape, true_scale=true_scale)
    design = rng.random((c['scenarios'], c['shifts'], c['machines']))
    actual = rng.random((c['replications'], c['shifts'], c['machines']))
    np.savez_compressed(out / 'uncertainty.npz', design=design, actual=actual)


def fit(c, out):
    d = np.load(out / 'lifetimes.npz')
    parameters = []
    for lifetime, event in zip(d['lifetime'], d['event']):
        def objective(theta):
            shape, scale = np.exp(theta)
            log_likelihood = event * (np.log(shape) - shape * np.log(scale) + (shape - 1) * np.log(lifetime)) - (lifetime / scale) ** shape
            return -log_likelihood.sum()
        result = minimize(objective, np.log([2., np.median(lifetime)]), method='L-BFGS-B', bounds=[(-2, 2), (0, 8)])
        parameters.append(np.exp(result.x))
    np.save(out / 'prognostics.npy', np.array(parameters))


def build(jobs, completed, state, pm, order, start_shift, c):
    remaining = {j['id']: j for j in jobs if j['id'] not in completed}
    ready = np.array(state['ready'], dtype=float).copy()
    family = np.array(state['family'], dtype=int).copy()
    activities = []
    kit_returns = list(state.get('kit_returns', []))
    for shift in range(start_shift, c['shifts']):
        start, end = shift * c['shift_hours'], (shift + 1) * c['shift_hours']
        kit_returns = [time for time in kit_returns if time > start]
        maintenance = np.flatnonzero(pm == shift)
        if len(maintenance) > min(c['technicians'], 2) or len(kit_returns) + len(maintenance) > c['kits']:
            return None
        for machine in maintenance:
            if ready[machine] > start:
                return None
            activities.append(dict(kind='pm', shift=shift, machine=int(machine), start=start, end=start + c['pm_hours'], job=-1))
            ready[machine] = start + c['pm_hours']
            kit_returns.append(start + c['kit_lead_shifts'] * c['shift_hours'])
        for job_id in order:
            if job_id not in remaining:
                continue
            job = remaining[job_id]
            choices = []
            for machine in np.flatnonzero(job['eligible']):
                setup = c['setup_hours'] * (family[machine] != job['family'])
                began = max(start, ready[machine], job['release'])
                finish = began + setup + job['duration'][machine]
                if finish <= end:
                    cost = finish + c['tardy_priority'] * max(0, finish - job['due']) * job['weight']
                    choices.append((cost, machine, began, finish, setup))
            if not choices:
                continue
            _, machine, began, finish, setup = min(choices)
            activities.append(dict(kind='production', shift=shift, machine=int(machine), start=began, end=finish, job=job_id, setup=setup))
            ready[machine], family[machine] = finish, job['family']
            remaining.pop(job_id)
    return activities


def replay(plan, jobs, state, parameters, uniforms, start_shift, c, only_first=False):
    ready = np.array(state['ready'], dtype=float).copy()
    age = np.array(state['age'], dtype=float).copy()
    family = np.array(state['family'], dtype=int).copy()
    completed = set()
    totals = dict(cost=0., downtime=0., tardy_jobs=0, job_moves=0, maintenance_hours=0., electricity=0., production_hours=0.,
                  greenhouse_gas=0., material_loss=0., overtime=0., produced=0, kits=0)
    items = {j['id']: j for j in jobs}
    selected = [a for a in plan if not only_first or a['shift'] == start_shift]
    repair_intervals = [(a['start'], a['end']) for a in selected if a['kind'] == 'pm']
    kit_returns = list(state.get('kit_returns', []))
    used_hazard = set()
    failure_time = {}
    for activity in sorted(selected, key=lambda a: (a['shift'], a['start'], a['machine'])):
        m, shift = activity['machine'], activity['shift']
        start = max(activity['start'], ready[m])
        if activity['kind'] == 'pm':
            age[m] = 0
            ready[m] = start + c['pm_hours']
            totals['cost'] += c['pm_cost']
            totals['maintenance_hours'] += c['pm_hours']
            totals['kits'] += 1
            kit_returns = [end for end in kit_returns if end > start]
            if len(kit_returns) >= c['kits']:
                raise RuntimeError('Spare-kit inventory exhausted at committed maintenance')
            kit_returns.append(start + c['kit_lead_shifts'] * c['shift_hours'])
            totals['electricity'] += c['idle_power_kw'] * c['pm_hours']
            continue
        job = items[activity['job']]
        duration = job['duration'][m] + c['setup_hours'] * (family[m] != job['family'])
        if only_first and start + duration > (shift + 1) * c['shift_hours']:
            continue
        key = shift, m
        if key not in failure_time:
            shape, scale = parameters[m]
            failure_time[key] = scale * ((age[m] / scale) ** shape - np.log(max(uniforms[shift, m], 1e-12))) ** (1 / shape) - age[m]
        failure = key not in used_hazard and duration >= failure_time[key]
        finish = start + duration
        if failure:
            failed_at = start + max(0., failure_time[key])
            repair_start = failed_at
            while True:
                boundaries = sorted({repair_start, repair_start + c['repair_hours'], *[v for interval in repair_intervals for v in interval if repair_start < v < repair_start + c['repair_hours']]})
                conflict = None
                for left, right in zip(boundaries, boundaries[1:]):
                    middle = (left + right) / 2
                    occupied = [(a, b) for a, b in repair_intervals if a <= middle < b]
                    if len(occupied) >= min(c['technicians'], 2):
                        conflict = min(b for _, b in occupied)
                        break
                if conflict is None:
                    break
                repair_start = conflict
            repair_end = repair_start + c['repair_hours']
            repair_intervals.append((repair_start, repair_end))
            downtime = repair_end - failed_at
            finish += downtime
            totals['cost'] += c['repair_cost'] + c['downtime_cost'] * downtime + c['scrap_cost']
            totals['downtime'] += downtime
            totals['material_loss'] += c['scrap_kg']
            age[m] = 0
            used_hazard.add(key)
        else:
            failure_time[key] -= duration
        age[m] += duration
        ready[m], family[m] = finish, job['family']
        tardy = max(0., finish - job['due'])
        totals['cost'] += tardy * job['weight'] * c['tardiness_cost']
        totals['tardy_jobs'] += int(tardy > 0)
        totals['electricity'] += duration * c['production_power_kw']
        totals['production_hours'] += duration
        previous = state.get('previous_assignment', {}).get(str(job['id']))
        if previous is not None and (previous[0] != m or abs(previous[1] - activity['start']) > c['shift_hours']):
            totals['job_moves'] += 1
            totals['cost'] += c['plan_change_cost']
        totals['overtime'] += max(0., finish - (shift + 1) * c['shift_hours'])
        totals['produced'] += 1
        completed.add(job['id'])
    totals['greenhouse_gas'] = totals['electricity'] * c['grid_kg_per_kwh'] + totals['kits'] * c['kit_kg_co2'] + totals['material_loss'] * c['scrap_kg_co2']
    totals['cost'] += totals['electricity'] * c['electricity_price']
    return totals, dict(ready=ready.tolist(), age=age.tolist(), family=family.tolist(), kit_returns=kit_returns), completed


def optimize(jobs, completed, state, parameters, scenarios, shift, c, rng, method, focal=None):
    ids = [j['id'] for j in sorted(jobs, key=lambda j: (j['due'], j['id'])) if j['id'] not in completed]
    choices = []
    for trial in range(c['candidate_plans']):
        pm = np.full(c['machines'], -1, dtype=int)
        age = np.array(state['age'])
        risk = 1 - np.exp(-((age + c['shift_hours']) / parameters[:, 1]) ** parameters[:, 0] + (age / parameters[:, 1]) ** parameters[:, 0])
        ranked = np.argsort(-risk)
        for m in ranked[:min(c['technicians'], 2, c['kits'])]:
            if method == 'reactive':
                continue
            if method == 'periodic':
                pm[m] = shift if age[m] >= c['periodic_hours'] else -1
            elif method == 'two-stage':
                pm[m] = shift if risk[m] >= c['maintenance_threshold'] else -1
            else:
                pm[m] = -1 if trial == 0 else [shift, shift + 1, -1][int(rng.integers(3))]
        if focal is not None:
            m, action = focal
            pm[m] = {'now': shift, 'next': shift + 1, 'continue': -1}[action]
        if (pm >= c['shifts']).any():
            continue
        order = ids.copy()
        if trial and method not in ['two-stage', 'periodic', 'reactive']:
            keys = rng.normal(0, c['priority_noise'], len(order))
            order = [a for _, a in sorted(zip(np.arange(len(order)) + keys, order))]
        plan = build(jobs, completed, state, pm, order, shift, c)
        if plan is None:
            continue
        outcomes = [replay(plan, jobs, state, parameters, sample, shift, c)[0] for sample in scenarios]
        cost = np.array([r['cost'] for r in outcomes])
        unscheduled = c['jobs'] - len(completed) - outcomes[0]['produced']
        score = cost.mean() + (0 if method in ['no-risk', 'point', 'two-stage'] else c['risk_weight']) * cvar(cost) + c['unfinished_cost'] * unscheduled
        choices.append((score, plan, outcomes))
    if not choices:
        return None
    return min(choices, key=lambda result: result[0])


def run(c, out):
    jobs = json.loads((out / 'instance.json').read_text(encoding='utf-8'))['jobs']
    fitted = np.load(out / 'prognostics.npy')
    uncertainty = np.load(out / 'uncertainty.npz')
    lifetimes = np.load(out / 'lifetimes.npz')
    true_parameters = np.column_stack([lifetimes['true_shape'], lifetimes['true_scale']])
    results, cards = [], []
    for replication in range(c['replications']):
        for method in ['MaintPlanAI', 'no-risk', 'point', 'two-stage', 'periodic', 'reactive']:
            rng = np.random.default_rng(c['seed'] + replication)
            state = dict(ready=[0.] * c['machines'], age=[0.] * c['machines'], family=[-1] * c['machines'], kit_returns=[], previous_assignment={})
            completed = set()
            totals = []
            for shift in range(c['shifts']):
                scenarios = uncertainty['design']
                if method == 'point':
                    scenarios = np.full_like(scenarios[:1], .5)
                solution = optimize(jobs, completed, state, fitted, scenarios, shift, c, rng, method)
                if solution is None:
                    raise RuntimeError('No feasible rolling plan at shift ' + str(shift))
                if method == 'MaintPlanAI':
                    alternatives = []
                    focal = int(np.argmax(state['age']))
                    for action in ['now', 'next', 'continue']:
                        result = optimize(jobs, completed, state, fitted, scenarios, shift, c,
                                          np.random.default_rng(c['seed'] + shift), method, (focal, action))
                        if result is not None:
                            costs = [value['cost'] for value in result[2]]
                            card = dict(replication=replication, shift=shift, machine=focal, action=action,
                                        expected_cost=float(np.mean(costs)), cvar_cost=cvar(costs),
                                        downtime=np.mean([v['downtime'] for v in result[2]]),
                                        tardy_jobs=np.mean([v['tardy_jobs'] for v in result[2]]),
                                        maintenance_hours=np.mean([v['maintenance_hours'] for v in result[2]]),
                                        job_moves=np.mean([v['job_moves'] for v in result[2]]),
                                        electricity=np.mean([v['electricity'] for v in result[2]]),
                                        greenhouse_gas=np.mean([v['greenhouse_gas'] for v in result[2]]))
                            alternatives.append(card)
                    if alternatives:
                        minimum = min(a['expected_cost'] for a in alternatives)
                        for card in alternatives:
                            card['regret'] = card['expected_cost'] - minimum
                            card['tail_regret'] = card['cvar_cost'] - min(a['cvar_cost'] for a in alternatives)
                            ranked = sorted(a['expected_cost'] + c['risk_weight'] * a['cvar_cost'] for a in alternatives)
                            card['rank_gap'] = (ranked[1] - ranked[0]) / max(abs(ranked[0]), 1e-8) if len(ranked) > 1 else np.nan
                            reference = next((a for a in alternatives if a['action'] == 'continue'), None)
                            card['burden_transfer'] = bool(reference is not None and card['expected_cost'] < reference['expected_cost'] and any(card[key] > reference[key] * 1.05 for key in ['electricity','greenhouse_gas','tardy_jobs']))
                        cards.extend(alternatives)
                outcome, state, done = replay(solution[1], jobs, state, true_parameters,
                                               uncertainty['actual'][replication], shift, c, only_first=True)
                state['previous_assignment'] = {str(a['job']): [a['machine'], a['start']] for a in solution[1] if a['kind'] == 'production' and a['job'] not in done}
                completed |= done
                totals.append(outcome)
            result = {key: sum(t[key] for t in totals) for key in totals[0]}
            result.update(method=method, replication=replication, unfinished=c['jobs'] - len(completed))
            idle_energy = c['idle_power_kw'] * max(0., c['machines'] * c['shifts'] * c['shift_hours'] - result['production_hours'] - result['maintenance_hours'])
            result['electricity'] += idle_energy
            result['cost'] += idle_energy * c['electricity_price'] + result['unfinished'] * c['unfinished_cost']
            result['greenhouse_gas'] = result['electricity'] * c['grid_kg_per_kwh'] + result['kits'] * c['kit_kg_co2'] + result['material_loss'] * c['scrap_kg_co2']
            results.append(result)
    pd.DataFrame(results).to_csv(out / 'replications.csv', index=False)
    pd.DataFrame(cards).to_csv(out / 'action_cards.csv', index=False)


def evaluate(c, out):
    d = pd.read_csv(out / 'replications.csv')
    rows = []
    for method, g in d.groupby('method'):
        result = g.mean(numeric_only=True).to_dict()
        result.update(method=method, cvar_cost=cvar(g.cost), tardy_share=float((g.tardy_jobs / g.produced.clip(lower=1)).mean()),
                      technician_utilization=float(g.maintenance_hours.mean() / (c['shifts'] * c['shift_hours'] * c['technicians'])))
        rows.append(result)
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    lifetimes = np.load(out / 'lifetimes.npz')
    fitted = np.load(out / 'prognostics.npy')
    calibration = []
    rng = np.random.default_rng(c['seed'] + 10000)
    for horizon in c['calibration_horizons']:
        predicted, observed = [], []
        for m, (shape, scale) in enumerate(fitted):
            truth = weibull_min.rvs(lifetimes['true_shape'][m], scale=lifetimes['true_scale'][m], size=200, random_state=rng)
            predicted.extend([weibull_min.cdf(horizon, shape, scale=scale)] * len(truth))
            observed.extend(truth <= horizon)
        predicted, observed = np.array(predicted), np.array(observed)
        ece = sum(len(ids) / len(predicted) * abs(predicted[ids].mean() - observed[ids].mean()) for ids in np.array_split(np.argsort(predicted), 10))
        calibration.append(dict(horizon=horizon, ece=ece, brier=float(((predicted - observed) ** 2).mean())))
    pd.DataFrame(calibration).to_csv(out / 'prognostic_metrics.csv', index=False)


def plot(out):
    d = pd.read_csv(out / 'metrics.csv')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(d.cost, d.cvar_cost)
    for r in d.itertuples():
        ax.annotate(r.method, (r.cost, r.cvar_cost))
    ax.set(xlabel='Expected operational cost', ylabel='CVaR operational cost')
    fig.tight_layout()
    fig.savefig(out / 'risk_cost.png', dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['prepare', 'fit', 'run', 'evaluate', 'plot'])
    p.add_argument('--config', default='config.json')
    p.add_argument('--output', default='runs/default')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    settings = out / 'settings.json'
    settings.write_text(json.dumps(c, indent=2))
    if args.command == 'plot':
        plot(out)
    else:
        globals()[args.command](c, out)


if __name__ == '__main__':
    main()
