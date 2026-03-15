def lorenz_system(t, state):
    sigma, ro, betta = 10, 28, 8 / 3    # coefficients
    x1, x2, x3 = state
    return [
        sigma * (x2 - x1),
        x1 * (ro - x3) - x2,
        x1 * x2 - betta * x3
    ]
