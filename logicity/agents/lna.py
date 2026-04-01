"""
Local Navigation Assistant (LNA) Module
========================================

Implements a grid of Local Navigation Assistants placed at intersection
positions across the simulation map.  Each LNA covers a non-overlapping
rectangular zone and facilitates two-way communication between cars:

  1. Uplink:   cars transmit their top-k1 FOV entities to their LNA.
  2. Downlink: the LNA aggregates, filters, re-grounds, and selects
               top-k2 entities to relay back to each ego car.

Grid layout
-----------
- First LNA at (6, 6), subsequent ones spaced 46 cells apart in both
  x and y directions.
- Coverage zones tile the map with no overlap.  Boundaries fall at
  the midpoint between adjacent LNA centres.

Only **cars** participate in this scheme (pedestrians do not transmit
or receive), but pedestrian entities observed by cars can be relayed.
"""

import logging

logger = logging.getLogger(__name__)


class LocalNavigationAssistant:
    """Manages LNA grid placement, coverage areas, and position look-ups."""

    LNA_START = 6        # first LNA centre coordinate
    LNA_SPACING = 46     # distance between adjacent LNA centres

    def __init__(self, world_size, agent_region=None):
        """
        Build the LNA grid for a square world of *world_size* cells.

        Args:
            world_size (int): Side length of the simulation grid (e.g. 241).
            agent_region (int | None): If set, only LNAs whose coverage
                intersects [0, agent_region) are considered active.
        """
        self.world_size = world_size
        self.agent_region = agent_region

        centres_1d = []
        c = self.LNA_START
        while c < world_size:
            centres_1d.append(c)
            c += self.LNA_SPACING

        self.centres_1d = centres_1d
        self.n_per_axis = len(centres_1d)

        self.lnas = []
        self._coverage = []
        for ix, cx in enumerate(centres_1d):
            for iy, cy in enumerate(centres_1d):
                x_min, x_max = self._axis_bounds(ix, centres_1d, world_size)
                y_min, y_max = self._axis_bounds(iy, centres_1d, world_size)
                entry = {
                    'id': len(self.lnas),
                    'centre': (cx, cy),
                    'x_min': x_min, 'x_max': x_max,
                    'y_min': y_min, 'y_max': y_max,
                }
                self.lnas.append(entry)
                self._coverage.append(entry)

        if agent_region is not None:
            self.active_lnas = [
                lna for lna in self.lnas
                if lna['x_min'] < agent_region and lna['y_min'] < agent_region
            ]
        else:
            self.active_lnas = list(self.lnas)

        logger.info(
            f"LNA grid: {self.n_per_axis}x{self.n_per_axis} = "
            f"{len(self.lnas)} total, {len(self.active_lnas)} active "
            f"(agent_region={agent_region})"
        )
        for lna in self.active_lnas:
            logger.debug(
                f"  LNA {lna['id']} centre={lna['centre']}  "
                f"coverage=[{lna['x_min']},{lna['x_max']})x"
                f"[{lna['y_min']},{lna['y_max']})"
            )

    @staticmethod
    def _axis_bounds(idx, centres, world_size):
        """Non-overlapping coverage interval along one axis."""
        if idx == 0:
            lo = 0
        else:
            lo = (centres[idx - 1] + centres[idx]) // 2 + 1

        if idx == len(centres) - 1:
            hi = world_size
        else:
            hi = (centres[idx] + centres[idx + 1]) // 2 + 1

        return lo, hi

    def get_lna_for_position(self, x, y):
        """
        Return the LNA dict that covers position (x, y), or None if
        the position falls outside all active LNA zones.
        """
        for lna in self.active_lnas:
            if (lna['x_min'] <= x < lna['x_max'] and
                    lna['y_min'] <= y < lna['y_max']):
                return lna
        return None

    def assign_cars_to_lnas(self, car_positions):
        """
        Map each car to its LNA.

        Args:
            car_positions: dict  {car_id: (x, y)}

        Returns:
            dict  {lna_id: [car_id, ...]}   (only active LNAs)
        """
        lna_to_cars = {lna['id']: [] for lna in self.active_lnas}
        for car_id, (x, y) in car_positions.items():
            lna = self.get_lna_for_position(x, y)
            if lna is not None:
                lna_to_cars[lna['id']].append(car_id)
        return lna_to_cars
