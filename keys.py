import random

import tweepy as tweepy

# Create keys
def get_keys():
    oauth_keys = [
        ["dfc6Wf3dTENuTlXDV1f9hml6B", "50vjNuiW6V5rZSRWvCyZy9RSY4h134Y9CjHHUtaxKbzqCdnn6D",
         "1359361622-LwyudniQqFogVrRXwoerQCSugRmU95nYKpWPrfS", "4nSO0bMINyisgrRfZgOD4SbAUOVcgC7BRq1N9j2AdTq5U"],
        ["lZkQXF2uZgljsr84A9ZnToFrS", "1fSeS1NRnSJW5rLV3snOg2NJOAVSzymKBumEbS40Lo4cge1Hwu",
         "1299631912659623936-ti2P7XgEfZkBdoOwlveFILCTOgShKG", "COffmtis6f3hbTyxbKOrNDfOUAygjoubpex4ytaXXsJzo"],
        ["lUwGGzNTUOiO16pzmxCLFrgCj", "FLH2utmpJ9y0HLQDcdlKKeGl4ZheY6s7osa3QzUGSYJGL1TkWt",
         "1299630171419533313-Knb2DgpfxEhENpqKkGvaTGXDwOFMhQ", "q0UvNWfxccp9AuCjGuxpBbBKbvSsywZ6Cbsk3hVDzhE1X"]
    ]

    auths = []
    for consumer_key, consumer_secret, access_key, access_secret in oauth_keys:
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        auths.append(auth)

    # return keys
    return random.choice(auths)