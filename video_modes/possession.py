class PossessionTracker:
    def __init__(self):
        self.team_counts = [0, 0]
        self.total = 0
    def update(self, team_id):
        if team_id in [0, 1]:
            self.team_counts[team_id] += 1
            self.total += 1
    def get_percentages(self):
        if self.total == 0:
            return (0.0, 0.0)
        return (100 * self.team_counts[0] / self.total, 100 * self.team_counts[1] / self.total) 