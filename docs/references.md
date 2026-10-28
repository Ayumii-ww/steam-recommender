# References and background notes

Sources used to motivate modelling choices. Each one was checked against the original paper or its official record.

## Why hours played is not a pure enjoyment signal

**Finding in our data** (`notebooks/01_eda.ipynb`, section 8): the share of thumbs-down reviews falls from about 45% for reviews written under 1 hour of play to about 9% at 50–100 hours, then rises again to about 13% beyond 600 hours. The rise is strongest for free-to-play games and also appears for games priced $20 or more; cheap paid games stay flat.

**Draft README paragraph:**

> Long playtime does not always mean a player enjoys a game. In our data, the share of negative reviews falls from 45% (under 1 hour) to 9% (50–100 hours), then rises to 13% beyond 600 hours, most strongly for free-to-play games. This pattern is consistent with the sunk cost effect, the tendency to keep going once money, time or effort has been invested (Arkes & Blumer, 1985), and with research linking obsessive passion for online games to more time played and more negative feelings while playing (Lafrenière et al., 2009). A survey of Call of Duty players likewise found that negative game experience rose with total playtime and that in-game spending rose with time played (Kloska, 2021). Our data shows a correlation, not a cause: we cannot see in-game spending or players' reasons. It motivated testing lower confidence for negative reviews in the ALS model: halving their confidence made no meaningful difference (Recall@10 0.0827 vs 0.0831), while leaving them out entirely lowered it by about 5% (0.0789), so negative reviews still carry useful signal about a player's tastes.

## Sources

**Arkes, H. R., & Blumer, C. (1985).** The psychology of sunk cost. *Organizational Behavior and Human Decision Processes, 35*(1), 124–140.
- Defines the sunk cost effect as a stronger urge to continue something after money, effort or time has already been invested.
- Field study: theater season-ticket buyers who paid full price used more tickets in the first half of the season (4.11 plays) than buyers given a $2 or $7 discount (3.32 and 3.29). The effect faded in the second half.
- Use for: the general concept, and that time as well as money counts as an investment.

**Lafrenière, M.-A. K., Vallerand, R. J., Donahue, E. G., & Lavigne, G. L. (2009).** On the costs and benefits of gaming: The role of passion. *CyberPsychology & Behavior, 12*(3), 285–290. https://doi.org/10.1089/cpb.2008.0234 (PMID 19366320)
- 222 players of massively multiplayer online games, using the dualistic model of passion.
- Both harmonious and obsessive passion were linked to positive feelings while playing, but only obsessive passion was linked to negative feelings, problematic behaviour, more time spent playing and physical symptoms.
- Use for: why some players keep playing long hours while feeling negatively about it.

**Kloska, M. M. (2021).** *Microtransactions and game enjoyment in free-to-play versus paid computer games* [Master's thesis, Tilburg University]. Supervisor: C. Kogler. https://arno.uvt.nl/show.cgi?fid=156773
- Survey of 317 Call of Duty: Modern Warfare / Warzone players.
- Money spent in-game correlated with total time played (r = .129, p = .047). Negative game experience rose with total time played. Free players spent more on microtransactions than players who bought the game.
- The author notes the sunk cost explanation fits time spent, but not game ownership.
- Use for: the link between playtime, in-game spending and negative experience. Cite as a master's thesis (not peer-reviewed); small, self-selected sample from Facebook groups.

## Not cited

AI-generated search summaries listed several further papers (dated 2026 on ResearchGate, or from non-indexed outlets). They could not be verified against a journal or official record, so they are not used.
