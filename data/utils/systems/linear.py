def linear_system(t, state):
    x1, x2 = state
    return [
        -0.1 * x1 + 3 * x2,
        -3 * x1 - 0.1 * x2
    ]
