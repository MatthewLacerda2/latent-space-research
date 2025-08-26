This is to visualize the change in cosine similarity between the description of states, as they progress

We only used nomic-embed-text for the embeddings, at default dimension count

# Example prompts:

Beginning: an 18yo slim beginner muay thai student. he is slim, not athletic and polite
End: a 20yo athletic guy in his first professional muay thai fight. he is athletic, educated but confident

Beginning: a 600 elo beginner chess player who plays the game for a lot of fun, hates losing, never spots the tactics, blunders a lot
End: a 2000 elo chess player who only plays competitively, seen every type of match, always spot tactics 2-3 moves ahead, rarely ever blunders

Beginning: a thunderstorm day with strong winds, lots of rain, very cloudy barely any sun light, when you can only hear the strong wind and the raindrops
End: a sunny day with clear skies, slight breeze, tropical climate, heartwarming

# Findings

The similarity between the N-th state and the end state grew as the states progressed

Output quality: better prompt and model yield better outputs more aligned with what we expected from progression states