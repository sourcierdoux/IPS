class beacon:
    def __init__(self, x, y, address, name, power_ref):
        self.x = x
        self.y = y
        self.address = address
        self.name = name
        self.power_ref = power_ref
    d_to_user=0
    d_2D=0
    z=0


class Person:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y