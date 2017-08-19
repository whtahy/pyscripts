import randomstate.prng.pcg64 as rng


def roll(dice):
    pivot = dice.find('d')
    num_dice = int(dice[0:pivot])
    num_sides = int(dice[pivot + 1: len(dice)])
    return num_dice + sum(rng.randint(0, num_sides, num_dice))


def seed(number):
    rng.seed(number)


def get_state():
    return rng.get_state()


def set_state(state):
    rng.set_state(state)


def get_int(a, b):
    return get_ints(a,b)


def get_ints(a, b, nints=1):
    return rng.randint(a, b + 1, nints)


def flip(success=50):
    return get_int(1, 100) <= success
