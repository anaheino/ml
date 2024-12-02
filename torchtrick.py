import torch
from torch.nn import functional as F


B, T, C = 4, 8, 2 # c = channels, dimensions
xbow = torch.zeros((B, T, C))
x = torch.randn(B, T, C)
# random tensori, kokoa 4,8,2, esim tensor([[[ 1.4505,  0.1375],
#         [-2.2255, -1.4001],
#         [ 0.2307,  0.6126],
#         [-1.3439,  0.6426],
#         [-1.4704,  1.1106],
#         [-0.9845, -0.4914],
#         [ 0.7618, -0.8306],
#         [ 1.0764,  2.0566]],

#        [[-0.0989,  0.1083],
#         [-1.0792, -0.0480],
#         [-0.6397,  1.0320],
#         [ 0.6693,  0.6308],
#         [ 0.2950, -1.3970],
#         [-0.6454, -0.1648],
#         [-0.2652,  0.5543],
#         [-0.3466,  0.0248]],

#        [[ 0.8716,  0.8802],
#         [-1.5755, -3.2416],
#         [ 0.5988, -1.2174],
#         [-0.8893, -0.2930],
#         [ 0.4191,  0.9988],
#         [ 0.2450, -3.7733],
#         [-1.5737,  0.1479],
#         [ 2.0603,  0.9507]],

#        [[-1.0898, -0.4492],
#         [ 0.8536, -0.5270],
#         [ 1.0134,  0.2712],
#         [ 0.5867, -1.8570],
#         [ 0.6594,  0.7984],
#         [-0.6735,  1.2327],
#         [ 1.1038,  1.4318],
#         [-0.7175, -0.4687]]])

# HUOM! Batch size === SLAISSI, eli eka dimensio,
# HUOM! 8 === contextin koko
# HUOM! 2 === olemassaoleva (nykyinen) + 1
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t, C), in this batch, take all up to this + 1 (time)
        xbow[b,t] = torch.mean(xprev, 0) # x bag of words, keskiarvo aiemmista, 0 dim, eli pelkkä luku
        # eli lopputuloksena on matriisi, jossa jokainen arvo vaihdetaan keskiarvolla jokaisesta aiemmasta arvosta, näin interaktioidaan
        #  aiempien kerroksien tai arvojen kanssa (naiivi tapa)

# is the same as
# short for weights
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (T, T) @ (B, T, C) -> (B, T, C)
print(xbow)
print(xbow2)
print(torch.allclose(xbow, xbow2))


#3 version

# esimerkit nelosella, vaikka oikeasti tehdään 8:lla
# [1, 1, 1, 1]
# [1, 1, 1, 1]
# [1, 1, 1, 1]
# [1, 1, 1, 1]
tril = torch.tril(torch.ones(T,T)) # tarkoittaa -> torch, tee mulle tämmöinen: 8x8 matriisi, jossa on kaikissa 1
# tril = all elements above the specified diagonal are zeroed out, eli tästä tulee halkaistu matriisi, joka on: 
# [1, 0, 0, 0]
# [1, 1, 0, 0]
# [1, 1, 1, 0]
# [1, 1, 1, 1]
neuwei = torch.zeros((T,T)) # tarkoittaa -> torch, tee mulle tämmöinen: 8x8 matriisi, jossa kaikissa on 0
# kaikki alkaa 0:na, KOSKA, me saadaan tietoon monta tokenia menneisyydestä otetaan huomioon! eli toisinsanoen eka 0, sitten 1, sitten 2....
# [0, 0, 0, 0]
# [0, 0, 0, 0]
# [0, 0, 0, 0]
# [0, 0, 0, 0]
neuwei = neuwei.masked_fill=(tril == 0, float('-inf')) 
# käytännössä, menneisyyden tokenit ei voi kommunikoida, eli toisinsanoen ne asetetaan negatiiviseks 
# tää tarkoittaa = muuta tota neuwei matriisia, että kaikki nollat on -inf, jos ne on 0 trillissä:
# [0, -inf, -inf, -inf]
# [0,   0, - inf, -inf]
# [0,   0,   0,   -inf]
# [0,   0,   0,     0 ]
neuwei= F.softmax(neuwei, dim=-1) # eli muuta nämä seuraavaan muotoon:
# [1,    0,    0,    0   ]
# [0.5,  0.5,  0,    0   ]
# [0.33, 0.33, 0.33, 0   ]
# [0.25, 0.25, 0.25, 0.25]
#....

# kun tämä matriisi kerrotaan alkumatriisilla, saadaan tasan sama tulos, kuin olisi saatu käymällä sisäikkäisillä
# luupeilla läpi tuo alkuperäinen sisäkkäisten luuppien arvo, eli tämä on se varsinainen aggregaatio, jolla lasketaan paljon menneisyyden
# tokeneille annetaan arvoa, eli tällä tavalla löydetään esim the -> the man, man katsoo "the"tä
xbow3 = neuwei @ x 
torch.allclose(xbow, xbow3)