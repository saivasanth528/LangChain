from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Space is a boundless expanse that stretches far beyond our skies. It begins just above Earth’s atmosphere and 
extends for billions of light-years, containing stars, planets, galaxies, black holes, and countless other celestial
wonders. Though it appears silent and empty, space is a dynamic, ever-changing place. Stars are born in dense clouds
of gas and dust, while others explode in brilliant supernovae. Entire galaxies collide and merge in cosmic dances 
that take billions of years.

Despite humanity's progress in exploring the cosmos—from landing on the Moon to sending probes beyond our solar 
system—we’ve only just begun to scratch the surface of what lies beyond. Space challenges our understanding of time,
gravity, and the very nature of existence, making it one of the greatest frontiers left to explore.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)
chunks = splitter.split_text(text)

print(len(chunks))

print(chunks)