import selfies

def selfies_to_smiles(sf):
    try:
        smiles = selfies.decoder(sf)
    except:
        smiles = ""
    return smiles
        