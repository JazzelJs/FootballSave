"""Pitch model: real-world positions of pitch markings, in meters.

Frame (CLAUDE.md): origin = centre of the goal line of the attacked goal,
x along the goal line (+x = goalkeeper's right), y into the field, z up.
"left"/"right" below always mean the GOALKEEPER's left/right, never the screen's.

Run `uv run python src/calib/pitch_model.py` to check your numbers.
"""
import math

# Standard measurements, same in every stadium (meters).
BOX_WIDTH = 40.32         # penalty box, along the goal line
BOX_DEPTH = 16.5          # penalty box, into the field
GOAL_AREA_WIDTH = 18.32
GOAL_AREA_DEPTH = 5.5
GOAL_WIDTH = 7.32         # between the posts
PENALTY_SPOT = 11.0       # distance from the goal line
ARC_RADIUS = 9.15         # penalty arc, centred on the penalty spot

# Sideways distance from the centre to where the arc crosses the box's far line.
# Right triangle: hypotenuse = ARC_RADIUS, one leg = BOX_DEPTH - PENALTY_SPOT (5.5 m).
ARC_X = math.sqrt(ARC_RADIUS**2 - (BOX_DEPTH - PENALTY_SPOT)**2)

# name -> (x, y, z). Everything lies on the grass, so z = 0.
POINTS = {
    # Goal posts (where each post meets the goal line)
    "goalpost_left": (-GOAL_WIDTH / 2, 0.0, 0.0),
    "goalpost_right": (GOAL_WIDTH / 2, 0.0, 0.0),
    # Goal area: "goalline" corners sit ON the goal line, "front" corners on the edge
    # in front of the goalkeeper (facing the field)
    "goal_area_goalline_left": (-GOAL_AREA_WIDTH / 2, 0.0, 0.0),
    "goal_area_goalline_right": (GOAL_AREA_WIDTH / 2, 0.0, 0.0),
    "goal_area_front_left": (-GOAL_AREA_WIDTH / 2, GOAL_AREA_DEPTH, 0.0),
    "goal_area_front_right": (GOAL_AREA_WIDTH / 2, GOAL_AREA_DEPTH, 0.0),
    # Penalty box: same idea, "goalline" = on the goal line, "front" = on the 16.5 m line
    "box_goalline_left": (-BOX_WIDTH / 2, 0.0, 0.0),
    "box_goalline_right": (BOX_WIDTH / 2, 0.0, 0.0),
    "box_front_left": (-BOX_WIDTH / 2, BOX_DEPTH, 0.0),
    "box_front_right": (BOX_WIDTH / 2, BOX_DEPTH, 0.0),
    # Penalty spot
    "penalty_spot": (0.0, PENALTY_SPOT, 0.0),
    # Where the penalty arc meets the box's front line (the 16.5 m line)
    "arc_left": (-ARC_X, BOX_DEPTH, 0.0),
    "arc_right": (ARC_X, BOX_DEPTH, 0.0),
}

# Straight painted lines, as (start point, end point) pairs from POINTS.
# Used to draw the pitch back onto a frame. (The curved arc is drawn separately.)
LINES = [
    #  7 lines in total: the goal line, the penalty box's other 3 sides,
    # and the goal area's 3 sides (its 4th side is part of the goal line).
    # Example format: ("goalpost_left", "goalpost_right"),

    ("goal_area_goalline_left", "goal_area_front_left"),
    ("goal_area_front_left", "goal_area_front_right"),
    ("goal_area_front_right", "goal_area_goalline_right"),
    ("box_goalline_left", "box_goalline_right"),
    ("box_goalline_left", "box_front_left"),
    ("box_front_left", "box_front_right"),
    ("box_front_right", "box_goalline_right"),
]


def check():
    missing = [k for k, v in POINTS.items() if v is None and not k.startswith("arc_")]
    if missing:
        print("Still TODO:", ", ".join(missing))
        return
    P = {k: v for k, v in POINTS.items() if v is not None}
    d = lambda a, b: math.dist(P[a][:2], P[b][:2])
    assert all(p[2] == 0 for p in P.values()), "everything is on the grass: z must be 0"
    for name in P:
        if name.endswith("_right"):
            left = name.replace("_right", "_left")
            assert P[name][0] > 0, f"{name}: goalkeeper's right must have x > 0"
            assert P[left] and math.isclose(P[left][0], -P[name][0]) and P[left][1] == P[name][1], \
                f"{left} should mirror {name}"
    assert math.isclose(d("goalpost_left", "goalpost_right"), GOAL_WIDTH), "goal width"
    assert math.isclose(d("goal_area_front_left", "goal_area_front_right"), GOAL_AREA_WIDTH), "goal area width"
    assert math.isclose(d("goal_area_goalline_right", "goal_area_front_right"), GOAL_AREA_DEPTH), "goal area depth"
    assert math.isclose(d("box_front_left", "box_front_right"), BOX_WIDTH), "box width"
    assert math.isclose(d("box_goalline_right", "box_front_right"), BOX_DEPTH), "box depth"
    assert math.isclose(math.dist(P["penalty_spot"][:2], (0, 0)), PENALTY_SPOT), "penalty spot distance"
    assert all(P[n][1] == 0 for n in P if "_goalline_" in n or n.startswith("goalpost")), "goal-line points need y = 0"
    if "arc_right" in P:
        assert math.isclose(d("penalty_spot", "arc_right"), ARC_RADIUS), "arc point must be 9.15 m from the spot"
        assert math.isclose(P["arc_right"][1], BOX_DEPTH), "arc point must lie on the box's front line"
    print(f"All {len(P)} points pass.")


if __name__ == "__main__":
    check()
