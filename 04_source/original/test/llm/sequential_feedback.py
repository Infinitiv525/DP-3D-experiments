"""Rebuild optimizer-visible history from committed results, never from test data."""
from configurator_contract_v3 import VERSION, validate_feedback


def build_feedback(shared_rows, search_export, measurements, epochs):
    history = []
    for row in shared_rows:
        history.append({'id': row['id'], 'parameters': {k: row[k] for k in ('learning_rate', 'weight_decay', 'pressure_weight', 'divergence_weight')},
                        'status': 'complete', 'S': row['S'], 'velocity_error': row['velocity_error'],
                        'pressure_error': row['pressure_error'], 'divergence': row['divergence'],
                        'training_seconds': row['training_gpu_hours'] * 3600, 'decision_seconds': 0.})
    for item in search_export['state']['attempts']:
        if item['phase'] != 'finished': continue
        identifier = 'a' + str(item['number']).zfill(2)
        measured = measurements.get(identifier)
        if item['status'] == 'complete':
            if measured is None or measured['S'] != item['S']: raise ValueError('Committed successful result must have matching measurements')
            metrics = {k: measured[k] for k in ('S', 'velocity_error', 'pressure_error', 'divergence')}
        else: metrics = {k: None for k in ('S', 'velocity_error', 'pressure_error', 'divergence')}
        history.append(dict(id=identifier, parameters=item['parameters'], status=item['status'],
                            training_seconds=item['training_seconds'], decision_seconds=item['decision_seconds'], **metrics))
    remaining = search_export['state']['spec']['attempt_limit'] - len(history)
    feedback = {'contract_version': VERSION, 'phase': 'technical_pilot', 'epochs_per_trial': epochs,
                'remaining_training_seconds': search_export['remaining_training_seconds'], 'remaining_attempts': remaining,
                'history': history}
    validate_feedback(feedback)
    return feedback
