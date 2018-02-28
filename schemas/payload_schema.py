payload_schema = {
    'type': 'object',
    'properties': {
        'id': {'type': 'string'},
        's1': {'type': 'number'},
        's2': {'type': 'number'},
        's3': {'type': 'number'},
        's4': {'type': 'number'}
    },
    'additionalProperties': False,
    'required': ['id', 's1', 's2', 's3', 's4']
}