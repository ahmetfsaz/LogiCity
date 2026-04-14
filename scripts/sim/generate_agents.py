#!/usr/bin/env python3
"""Generate an agent YAML file with a configurable number of cars and pedestrians.

Usage:
    python3 generate_agents.py --cars 20 --peds 8 --output agents_20c_8p.yaml
"""
import argparse

CAR_CONCEPTS = [
    {"ambulance": 1.0},
    {"reckless": 1.0},
    {},
    {"tiro": 1.0},
    {"bus": 1.0},
    {"police": 1.0},
]

PED_CONCEPTS = [
    {"young": 1.0},
    {"young": 1.0},
    {"old": 1.0},
]


def generate(n_cars: int, n_peds: int) -> str:
    lines = [
        "# Auto-generated agent config",
        "agents:",
    ]

    for i in range(1, n_cars + 1):
        extra = CAR_CONCEPTS[(i - 1) % len(CAR_CONCEPTS)]
        lines.append(f"- class: Private_car")
        lines.append(f"  id: {i}")
        lines.append(f"  size: 1")
        lines.append(f"  gplanner: A*vg")
        lines.append(f"  concepts:")
        lines.append(f"    type: Car")
        for k, v in extra.items():
            lines.append(f"    {k}: {v}")
        lines.append(f"    priority: {i}")
        lines.append("")

    for i in range(1, n_peds + 1):
        extra = PED_CONCEPTS[(i - 1) % len(PED_CONCEPTS)]
        lines.append(f"- class: Pedestrian")
        lines.append(f"  id: {i}")
        lines.append(f"  size: 1")
        lines.append(f"  gplanner: A*")
        lines.append(f"  concepts:")
        lines.append(f"    type: Pedestrian")
        lines.append(f"    priority: 0")
        for k, v in extra.items():
            lines.append(f"    {k}: {v}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cars", type=int, required=True)
    parser.add_argument("--peds", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    content = generate(args.cars, args.peds)
    with open(args.output, "w") as f:
        f.write(content)
    import sys
    print(f"Generated {args.output}: {args.cars} cars, {args.peds} pedestrians", file=sys.stderr)
