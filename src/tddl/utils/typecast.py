from typing import get_type_hints
from pathlib import Path

SUPPORTED_TYPES = [str, float, int, Path, bool]

def typecast(func):
    typeof = get_type_hints(func)

    def inner(**kwargs):
        typed_kwargs = {
            key: typeof[key](val)
                if
                    typeof.get(key) is not None 
                and 
                    typeof[key] in SUPPORTED_TYPES 
                else val
            for key, val in kwargs.items()
        }
        return func(**typed_kwargs)
    return inner


if __name__ == "__main__":
    from typing import List

    def untyped_function(a: int, b: float, c: Path, d: List[int], **kwargs):
        print(a, type(a))
        print(b, type(b))
        print(c, type(c))
        print(d, type(d))
        print({k:type(v) for k,v in kwargs.items()})

    @typecast
    def typed_function(a: int, b: float, c: Path, d: List[int], **kwargs):
        print(a, type(a))
        print(b, type(b))
        print(c, type(c))
        print(d, type(d))
        print({k:type(v) for k,v in kwargs.items()})

    print(get_type_hints(untyped_function))
    
    untyped_function(a='1', b='1.2', c='foo', d='[1,2,3]', e=[1,2,3])
    typed_function(a='1', b='1.2', c='foo', d='[1,2,3]', e=[1,2,3])