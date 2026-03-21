# Standard Card Data

This folder stores imported data for the selected `ex Battle Deck` and `ex Deluxe Battle Deck` products that will eventually live under the `standard` game mode.

Current scope:

- only the deck list provided in the thread
- deck JSON split by product line:
  - `decks/ex-battle`
  - `decks/ex-deluxe-battle`
- `ex-battle` is the initial implementation target
- `ex-deluxe-battle` is imported and stored for later work only
- shared card catalog in `catalog`
- shared local artwork cache in `frontend/assets/cards/standard/shared`

Source split:

- Bulbapedia raw wiki pages for deck membership and release metadata
- `PokemonTCG/pokemon-tcg-data` for structured card fields and image URLs

No app integration is wired from this data yet.
