def armor_get(obj, key, default=None):
    """The ultimate safety net"""
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        elif hasattr(obj, '__getitem__'):
            return obj[key]
        return default
    except:
        return default
