import sys
import math
import random
from collections import Counter


# ============================================================
# Adaptive Goofspiel bot -- standard library only
# ============================================================
# Protocol expected:
#   GAME N open
#   PRIZES p1 ... pN
#   ROUND prize pot
#   -> BID x
#   RESULT opponent_bid my_points opponent_points
#   ...
#   GAME_END score


#
# Design:
#   * Bayesian-ish mixture of opponent bidding models.
#   * Learns opponent's bid percentile, not just raw card value.
#   * One-step simultaneous-action evaluation against the full
#     predicted distribution of opponent bids.
#   * Continuation/shadow-value model prevents wasting high cards.
#   * Carryover-aware: a tied pot is explicitly valued next round.
#   * Uses exact future prize order when open == 1.
#   * Uses the remaining prize multiset when open == 0.
#   * Small bounded randomization only when moves are near-equivalent.
#   * No third-party dependencies and no output except BID lines.


EPS = 1e-12


class GoofspielBot:
    MODEL_COUNT = 11

    def __init__(self, n, open_order, prizes):
        self.n = n
        self.open_order = bool(open_order)
        self.prizes = list(prizes)
        self.total_value = sum(prizes)
        self.max_prize = max(prizes) if prizes else 1
        self.min_prize = min(prizes) if prizes else 0

        self.my_hand = list(range(1, n + 1))
        self.opp_hand = list(range(1, n + 1))

        self.round_no = 0
        self.my_score = 0
        self.opp_score = 0

        self.unseen = Counter(prizes)
        self.known_pos = 0

        # Opponent-model weights.  Models intentionally overlap: the
        # posterior becomes robust rather than brittle after one observation.
        self.model_w = [1.0 / self.MODEL_COUNT] * self.MODEL_COUNT
        self.history = []
        self.error_ema = 0.34

        # Saved context for updating the opponent model after RESULT.
        self.pending = None
        self.last_bid = None
        self.last_prize = 0
        self.last_pot = 0

        # Randomization protects against history-learning opponents, but
        # only among actions whose estimated values are genuinely close.
        self.rng = random.Random()

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    @staticmethod
    def _clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    @staticmethod
    def _nearest_index_fraction(hand, q):
        m = len(hand)
        if m <= 1:
            return 0
        q = 0.0 if q < 0.0 else 1.0 if q > 1.0 else q
        return int(q * (m - 1) + 0.5)

    @staticmethod
    def _quantile(value, values):
        """Mid-rank quantile in [0,1], stable with duplicate prizes."""
        #WHAT THE FUCK AM I EVEN CODING DAWG THIS FUNCTIJN IS DAMN BRAINROT WHY WOULD I ANNOTATE IT TO THIS SHIT FUCK
        m = len(values)
        if m <= 1:
            return 0.5
        less = 0
        equal = 0
        for x in values:
            if x < value:
                less += 1
            elif x == value:
                equal += 1
        return (less + 0.5 * max(0, equal - 1)) / (m - 1)

    def _remaining_prizes_after_current(self):
        if self.open_order:
            # ROUND number r corresponds to prizes[r-1].  We still trust the
            # judge's actual current prize; this slice is only future info.
            return self.prizes[self.round_no:]

        out = []
        for value, count in self.unseen.items():
            if count > 0:
                out.extend([value] * count)
        return out

    def _context_features(self, prize, pot, opp_hand, future_prizes):
        all_prize_values = [prize] + list(future_prizes)
        all_stakes = [pot] + list(future_prizes)

        prize_q = self._quantile(prize, all_prize_values)
        stake_q = self._quantile(pot, all_stakes)

        # Normalized raw magnitudes complement rank-based models when prize
        # values are uneven (e.g. [0,0,1,2,100]).
        scale = max(1.0, float(max([pot] + all_prize_values)))
        prize_raw = prize / scale
        pot_raw = pot / scale

        progress = (self.round_no - 1) / max(1, self.n - 1)
        score_diff = (self.my_score - self.opp_score) / max(1.0, float(self.total_value))

        return {
            'prize': prize,
            'pot': pot,
            'prize_q': prize_q,
            'stake_q': stake_q,
            'prize_raw': prize_raw,
            'pot_raw': pot_raw,
            'progress': progress,
            'score_diff': score_diff,
            'opp_hand': tuple(opp_hand),
            'future': tuple(future_prizes),
        }

    # ------------------------------------------------------------
    # Opponent modelling
    # ------------------------------------------------------------
    def _history_knn_q(self, ctx):
        if not self.history:
            return ctx['stake_q']

        scored = []
        for h in self.history:
            # Pot/stake rank is the most predictive feature; prize rank and
            # game progress break ties between superficially similar rounds.
            d = (
                1.75 * abs(ctx['stake_q'] - h['stake_q'])
                + 0.90 * abs(ctx['prize_q'] - h['prize_q'])
                + 0.35 * abs(ctx['progress'] - h['progress'])
                + 0.20 * abs(ctx['score_diff'] - h['score_diff'])
            )
            scored.append((d, h['opp_q']))

        scored.sort(key=lambda z: z[0])
        k = min(4, len(scored))
        num = 0.0
        den = 0.0
        for d, q in scored[:k]:
            w = 1.0 / (0.07 + d)
            num += w * q
            den += w
        return num / den if den else ctx['stake_q']

    def _learned_offset_q(self, ctx):
        if not self.history:
            return ctx['stake_q']

        # Recent observations get slightly more weight.  Learning rank
        # residuals works even after high/low cards disappear from the hand.
        num = 0.0
        den = 0.0
        start = max(0, len(self.history) - 8)
        for j in range(start, len(self.history)):
            h = self.history[j]
            age = len(self.history) - 1 - j
            w = 0.82 ** age
            num += w * (h['opp_q'] - h['stake_q'])
            den += w
        offset = num / den if den else 0.0
        return self._clamp(ctx['stake_q'] + offset, 0.0, 1.0)

    def _model_qs(self, ctx):
        s = ctx['stake_q']
        p = ctx['prize_q']
        raw = ctx['prize_raw']

        # These cover common deterministic Goofspiel families plus two
        # online learners.  Predicting in percentile-space automatically
        # maps correctly into the opponent's shrinking hand.
        return [
            s,                                  # 0: match effective stake rank
            p,                                  # 1: match exposed prize rank
            self._clamp(s + 0.11, 0.0, 1.0),  # 2: stake + one-ish rank
            self._clamp(s - 0.11, 0.0, 1.0),  # 3: conservative
            s ** 0.72,                          # 4: aggressive mid/high
            s ** 1.42,                          # 5: save high cards
            self._clamp(0.65 * s + 0.35 * p, 0.0, 1.0),
            self._clamp(0.55 * p + 0.45 * raw, 0.0, 1.0),
            0.15 if s < 0.34 else (0.84 if s > 0.68 else 0.50),  # tiered
            self._history_knn_q(ctx),           # 9: local historical learner
            self._learned_offset_q(ctx),        # 10: learned aggressiveness
        ]

    def _predict_distribution(self, ctx):
        hand = list(ctx['opp_hand'])
        m = len(hand)
        if m == 1:
            return {hand[0]: 1.0}

        probs = {c: 0.0 for c in hand}
        qs = self._model_qs(ctx)

        # Spread each model around its modal card.  Real bots often snap a
        # desired bid to the nearest available card, so adjacent cards deserve
        # meaningful mass too.
        for w, q in zip(self.model_w, qs):
            idx = self._nearest_index_fraction(hand, q)
            kernel = ((0, 0.68), (-1, 0.14), (1, 0.14), (-2, 0.02), (2, 0.02))
            local_total = 0.0
            valid = []
            for d, kw in kernel:
                j = idx + d
                if 0 <= j < m:
                    valid.append((j, kw))
                    local_total += kw
            for j, kw in valid:
                probs[hand[j]] += w * kw / local_total

        # If observations look random/unmodelled, retain more uniform mass.
        # As a bot becomes predictable this shrinks toward 8%.
        uniform_mix = self._clamp(0.07 + 0.72 * self.error_ema, 0.08, 0.48)
        inv = 1.0 / m
        for c in hand:
            probs[c] = (1.0 - uniform_mix) * probs[c] + uniform_mix * inv

        total = sum(probs.values())
        if total <= 0.0:
            return {c: inv for c in hand}
        inv_total = 1.0 / total
        for c in hand:
            probs[c] *= inv_total
        return probs

    def _update_models(self, actual_bid):
        if self.pending is None:
            return

        ctx = self.pending['ctx']
        hand = list(ctx['opp_hand'])
        if actual_bid not in hand:
            return

        m = len(hand)
        actual_idx = hand.index(actual_bid)
        actual_q = 0.5 if m <= 1 else actual_idx / (m - 1)

        qs = self._model_qs(ctx)
        new_w = []
        avg_error = 0.0

        for w, q in zip(self.model_w, qs):
            pred_idx = self._nearest_index_fraction(hand, q)
            d = abs(pred_idx - actual_idx) / max(1, m - 1)
            avg_error += w * d

            # Broad likelihood prevents a single unusual bid from deleting a
            # model forever; exact/near-exact predictions still gain quickly.
            likelihood = 0.035 + math.exp(-0.5 * (d / 0.16) ** 2)
            new_w.append(w * likelihood)

        z = sum(new_w)
        if z <= EPS:
            self.model_w = [1.0 / self.MODEL_COUNT] * self.MODEL_COUNT
        else:
            # Tiny floor preserves model diversity against nonstationary bots.
            floor = 0.004
            tmp = [x / z for x in new_w]
            tmp = [max(floor, x) for x in tmp]
            z2 = sum(tmp)
            self.model_w = [x / z2 for x in tmp]

        self.error_ema = 0.68 * self.error_ema + 0.32 * avg_error

        self.history.append({
            'stake_q': ctx['stake_q'],
            'prize_q': ctx['prize_q'],
            'progress': ctx['progress'],
            'score_diff': ctx['score_diff'],
            'opp_q': actual_q,
            'opp_bid': actual_bid,
            'pot': ctx['pot'],
            'prize': ctx['prize'],
        })

    # ------------------------------------------------------------
    # Continuation / card shadow value
    # ------------------------------------------------------------
    @staticmethod
    def _remove_one(hand, card):
        # Hands are short (<=26); a copy is clearer and fast enough.
        out = list(hand)
        out.remove(card)
        return out

    def _monotone_edge(self, my_after, opp_after, stakes):
        """Approximate future score differential under efficient allocation.

        Sort future stakes and pair stronger bid cards with more valuable
        stakes.  The hand asymmetry created by today's bids becomes an
        explicit continuation value, so the bot does not burn a 26 to win a
        trivial pot unless that is actually worthwhile.
        """
        k = len(my_after)
        if k == 0 or not stakes:
            return 0.0

        a = sorted(my_after)
        b = sorted(opp_after)
        v = sorted(stakes)

        # Defensive if an unusual wrapper gives a mismatching prize count.
        k = min(len(a), len(b), len(v))
        a = a[-k:]
        b = b[-k:]
        v = v[-k:]

        edge = 0.0
        for x, y, stake in zip(a, b, v):
            if x > y:
                edge += stake
            elif x < y:
                edge -= stake
        return edge

    def _tie_future_edge_unknown(self, my_after, opp_after, future, carry):
        """Expected shadow value if today's bid ties in hidden-order play."""
        if not future:
            return 0.0

        counts = Counter(future)
        total = float(len(future))
        value = 0.0
        for nxt, cnt in counts.items():
            altered = list(future)
            altered.remove(nxt)
            altered.append(nxt + carry)
            value += (cnt / total) * self._monotone_edge(my_after, opp_after, altered)
        return value

    def _continuation_edge(self, my_after, opp_after, future, tied, carry):
        if not future:
            return 0.0

        if not tied:
            return self._monotone_edge(my_after, opp_after, future)

        if self.open_order:
            altered = list(future)
            altered[0] += carry
            return self._monotone_edge(my_after, opp_after, altered)

        return self._tie_future_edge_unknown(my_after, opp_after, future, carry)

    # ------------------------------------------------------------
    # Move selection
    # ------------------------------------------------------------
    def choose_bid(self, prize, pot):
        future = self._remaining_prizes_after_current()
        ctx = self._context_features(prize, pot, self.opp_hand, future)
        opp_dist = self._predict_distribution(ctx)

        # Last card is forced and bypasses all modelling.
        if len(self.my_hand) == 1:
            bid = self.my_hand[0]
            self.pending = {'ctx': ctx, 'bid': bid, 'values': {bid: 0.0}}
            self.last_bid = bid
            return bid

        current_diff = self.my_score - self.opp_score
        half_total = self.total_value / 2.0
        progress = (self.round_no - 1) / max(1, self.n - 1)

        # Trust continuation estimates more late in the game, when fewer
        # uncertain rounds remain.  Immediate pot value always stays exact.
        cont_discount = 0.58 + 0.34 * progress

        values = {}

        # Cache hypothetical post-bid hands.
        my_after_cache = {a: self._remove_one(self.my_hand, a) for a in self.my_hand}
        opp_after_cache = {b: self._remove_one(self.opp_hand, b) for b in self.opp_hand}

        for a in self.my_hand:
            ev = 0.0
            my_after = my_after_cache[a]

            for b, prob in opp_dist.items():
                opp_after = opp_after_cache[b]

                if a > b:
                    immediate = float(pot)
                    tied = False
                elif a < b:
                    immediate = -float(pot)
                    tied = False
                else:
                    immediate = 0.0
                    tied = True

                cont = self._continuation_edge(
                    my_after, opp_after, future, tied, pot
                )

                projected = current_diff + immediate + cont_discount * cont

                # Contest score is linear in final point differential until
                # +/- S/2, then clips.  Optimize in that same clipped space.
                if projected > half_total:
                    util = half_total
                elif projected < -half_total:
                    util = -half_total
                else:
                    util = projected

                ev += prob * util

            values[a] = ev

        ranked = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
        best_bid, best_value = ranked[0]

        # Bounded random play: only randomize when alternatives are close.
        # This is enough to make our mapping difficult to learn without
        # sacrificing a clearly superior move.
        if len(ranked) >= 2:
            uncertainty = self._clamp(self.error_ema, 0.0, 0.7)
            close_band = max(0.30, (0.010 + 0.025 * uncertainty) * max(1.0, self.total_value))
            near = [x for x in ranked[:4] if best_value - x[1] <= close_band]

            if len(near) > 1:
                # Higher uncertainty => slightly higher temperature.
                temp = max(0.18, 0.20 + 0.015 * self.total_value * uncertainty)
                weights = []
                for _, v in near:
                    weights.append(math.exp((v - best_value) / temp))

                r = self.rng.random() * sum(weights)
                acc = 0.0
                for (bid, _), w in zip(near, weights):
                    acc += w
                    if r <= acc:
                        best_bid = bid
                        break

        self.pending = {'ctx': ctx, 'bid': best_bid, 'values': values}
        self.last_bid = best_bid
        return best_bid

    # ------------------------------------------------------------
    # Judge events
    # ------------------------------------------------------------
    def on_round(self, prize, pot):
        self.round_no += 1
        self.last_prize = prize
        self.last_pot = pot

        if self.open_order:
            # known_pos isn't needed for slicing, but tracking it makes us
            # tolerant of duplicate prize values and protocol wrappers.
            self.known_pos += 1
        else:
            if self.unseen.get(prize, 0) > 0:
                self.unseen[prize] -= 1
                if self.unseen[prize] == 0:
                    del self.unseen[prize]

        return self.choose_bid(prize, pot)

    def on_result(self, opp_bid, my_points, opp_points):
        self._update_models(opp_bid)

        # Remove exactly the cards that were played.
        if self.last_bid in self.my_hand:
            self.my_hand.remove(self.last_bid)
        if opp_bid in self.opp_hand:
            self.opp_hand.remove(opp_bid)

        self.my_score = my_points
        self.opp_score = opp_points
        self.pending = None


# ============================================================
# Robust line protocol
# ============================================================

def read_nonempty_line():
    while True:
        line = sys.stdin.readline()
        if line == '':
            return None
        line = line.strip()
        if line:
            return line


def parse_game_header():
    line = read_nonempty_line()
    if line is None:
        return None
    parts = line.split()

    if parts[0] == 'GAME' and len(parts) >= 3:
        n = int(parts[1])
        open_order = int(parts[2])
    else:
        # Fallback: first two integer tokens are N and open.
        nums = []
        for token in parts:
            try:
                nums.append(int(token))
            except ValueError:
                pass
            if len(nums) == 2:
                break
        if len(nums) < 2:
            return None
        n, open_order = nums[0], nums[1]

    prizes = []
    while len(prizes) < n:
        line = read_nonempty_line()
        if line is None:
            return None
        p = line.split()
        start = 1 if p and p[0] == 'PRIZES' else 0
        for token in p[start:]:
            try:
                prizes.append(int(token))
            except ValueError:
                continue
            if len(prizes) == n:
                break

    return n, open_order, prizes


def main():
    init = parse_game_header()
    if init is None:
        return

    n, open_order, prizes = init
    bot = GoofspielBot(n, open_order, prizes)

    while True:
        line = sys.stdin.readline()
        if line == '':
            return
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == 'ROUND' and len(parts) >= 3:
            prize = int(parts[1])
            pot = int(parts[2])
            bid = bot.on_round(prize, pot)
            sys.stdout.write('BID ' + str(bid) + '\n')
            sys.stdout.flush()

        elif parts[0] == 'RESULT' and len(parts) >= 4:
            opp_bid = int(parts[1])
            my_points = int(parts[2])
            opp_points = int(parts[3])
            bot.on_result(opp_bid, my_points, opp_points)

        elif parts[0] == 'GAME_END':
            return

        # Ignore any wrapper/status messages. Never print except on ROUND.


if __name__ == '__main__':
    main()
