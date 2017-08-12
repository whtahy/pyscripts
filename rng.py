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
    try:
        return rng.randint(min(a, b), max(a, b) + 1, 1)[0]
    except:
        return a


def get_ints(a, b, number_of_ints=1):
    try:
        return rng.randint(min(a, b), max(a, b) + 1, number_of_ints)
    except:
        return a


def flip(success=50):
    return get_int(1, 100) <= success
