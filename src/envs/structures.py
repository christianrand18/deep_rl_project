class Passenger:
    """Passenger class"""

    def __init__(self, id, origin, destination, request_time, price, max_wait=2) -> None:
        self.id = id
        self.origin = origin
        self.destination = destination
        self.request_time = request_time
        self.price = price
        self.wait_time = 0
        self.max_wait = max_wait

    def unmatched_update(self):
        self.wait_time += 1
        return self.wait_time >= self.max_wait


def generate_passenger(demand, max_wait=2, arrivals=None):
    """
    Generate passenger according to the demand

    demand: (origin,destination,time,total demand,price)
    arrivals: number of passengers already arrive in the system

    return: list of new passengers, total number of passenger arrivals
    """
    newp = []
    ori, des, t, d, p = demand
    for i in range(d):
        if arrivals is None:
            newp.append(Passenger(i, ori, des, t, p, max_wait=max_wait))
        else:
            newp.append(Passenger(arrivals + 1, ori, des, t, p, max_wait=max_wait))
            arrivals += 1

    return newp, arrivals
