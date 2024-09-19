import selfies

def selfies_to_smiles(sf):
    sf = sf.replace('<bom>', '')
    sf = sf.replace('<eom>', '')
    try:
        smiles = selfies.decoder(sf)
    except:
        smiles = ""
    return smiles

def smiles_to_selfies(smi):
    try:
        sf = selfies.encoder(smi)
    except:
        sf = ""
    return sf