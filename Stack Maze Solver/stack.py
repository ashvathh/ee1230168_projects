class Empty(Exception):
    pass

class ArrayStack:

    def __init__(self) -> None:
        self.data=[]

    def __len__(self) -> int:
        return len(self.data)

    def is_empty(self) -> bool:
        return len(self.data)==0

    def push(self,e) -> None:
        self.data.append(e)

    def top(self):
        if self.is_empty():
            raise Empty("Stack is empty")
        return self.data[-1]

    def pop(self):
        if self.is_empty():
            raise Empty("Stack is empty")
        return self.data.pop()

