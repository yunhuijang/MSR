import enum
from rdkit import Chem
from itertools import product

# # for debugging
# os.chdir('./stgg/src')

PAD_TOKEN = "[pad]"
BOS_TOKEN = "[bos]"
EOS_TOKEN = "[eos]"
UNK_TOKEN = "<unk>"
PASS_TOKEN = "[pass]"
# TODO: need to change node tokens? (just get only input things, not canonicalized things..? (but inputs are also canonicalized..))
# TODO: change tokens (input에 있는 token들만 + dataset 분리?)
NODE_TOKENS = [
    # QM9
    "C", "F", "O", "N",
    # ZINC
    'Br', 'Cl', 'I', 'P', 'S', 
    '[C@@H]', '[C@@]', '[C@H]', '[C@]', '[CH-]', '[CH2-]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', 
    '[O+]', '[O-]', '[OH+]', '[P+]', '[P@@H]', '[P@@]', '[P@]', '[PH+]', '[PH2]', '[PH]', 
    '[S+]', '[S-]', '[S@@+]', '[S@@]', '[S@]', '[SH+]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', 
    'c', 'n', 'o', 's',
    # MOSES
    "[H]"
    
    # "Br","C","Cl","F","I","N","O","P","S",
    # "[C@@H]","[C@@]","[C@H]","[C@]","[CH-]","[CH2-]","[H]","[N+]",
    # "[N-]","[NH+]","[NH-]","[NH2+]","[NH3+]","[O+]","[O-]","[OH+]",
    # "[P]", "[P+]","[P@@H]","[P@@]","[P@]", "[P@H]", "[PH+]","[PH2]","[PH]","[S+]","[S-]","[S@@+]","[S@@]","[S@]", "[S@+]","[S]","[SH+]",
    # "[n+]","[n-]","[nH+]","[nH]","[o+]","[s+]","[C-]","[c-]","[cH-]", "[C]", "[SH]", "[CH]","[IH]", "[IH2]", "[I]", "[I+]", "[I-]",
    # "c","n","o","s"
    ]
BOND_TOKENS = ["-", "=", "#", "/"]

# multiset tokens
END_OF_COUNT_TOKEN = "[eoc]"
BEGIN_OF_COUNT_TOKEN = "[boc]"
COUNT_TOKENS = [END_OF_COUNT_TOKEN, BEGIN_OF_COUNT_TOKEN]
COUNT_TOKENS.extend([f"{i}_{j}" for i, j in product(range(len(NODE_TOKENS)+len(BOND_TOKENS)+1), range(1,35))])
# EDGE_COUNT_TOKENS = [f"{i}_{j}" for i, j in product(range(len(BOND_TOKENS)), range(1,4))]

# ring tokens
BEGIN_OF_RING_TOKEN = "[bor]"
END_OF_RING_TOKEN = "[eor]"
RING_TOKENS = [BEGIN_OF_RING_TOKEN, END_OF_RING_TOKEN]
# i: ring size, j: number of rings
RING_TOKENS.extend([f"R{i}_{j}" for i, j in product(range(1,25), range(1,9))])

# fragment tokens
BEGIN_OF_FRAG_TOKEN = "[bof]"
END_OF_FRAG_TOKEN = "[eof]"

FRAG_TOKENS_DICT = {}
# for dataset in ['qm9', 'zinc', 'moses', 'zinc_sub', 'moses_sub']:
#     with open(f'resource/data/{dataset}/fragment.txt', 'r') as f:
#         FRAG_TOKENS = [line.strip() for line in f.readlines()]
#     frag_tokens = [BEGIN_OF_FRAG_TOKEN, END_OF_FRAG_TOKEN] + FRAG_TOKENS
#     FRAG_TOKENS_DICT[dataset] = frag_tokens

# index tokens
BEGIN_OF_INDEX_TOKEN = "[boi]"
END_OF_INDEX_TOKEN = "[eoi]"
INDEX_TOKENS = [BEGIN_OF_INDEX_TOKEN, END_OF_INDEX_TOKEN]
# TODO: find maximum index
INDEX_TOKENS.extend([f"I{i}" for i in range(40)])


TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, *NODE_TOKENS, *BOND_TOKENS,
    "(",")","1","2","3","4","5","6","7","8","9","\\"
    # # valence tokens
    # "[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]",
    # # ring tokens
    # "[R]", "[NR]",
    # # implicit Hs
    # "[H0]", "[H1]", "[H2]", "[H3]", "[H4]", "[H5]"
]
TOKENS_SELFIES = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
# in QM9
'[#Branch1]', '[#Branch2]', '[#C]', '[#N]', 
'[/C]', '[/Cl]', '[/NH1+1]', '[/N]', '[/O-1]', '[/O]', '[/S-1]', 
'[/S]', '[=Branch1]', '[=Branch2]', '[=C]', '[=N+1]', '[=NH1+1]', 
'[=NH2+1]', '[=N]', '[=O]', '[=Ring1]', '[=Ring2]', '[=S@@]', 
'[=S]', '[Br]', '[Branch1]', '[Branch2]', '[C@@H1]', '[C@@]', 
'[C@H1]', '[C@]', '[C]', '[Cl]', '[F]', '[I]', '[N+1]', '[N-1]', 
'[NH1+1]', '[NH1-1]', '[NH1]', '[NH2+1]', '[NH3+1]', '[N]', '[O-1]', 
'[O]', '[P@@]', '[P]', '[Ring1]', '[Ring2]', '[S-1]', '[S@@]', '[S@]', 
'[S]', '[\\C]', '[\\N]', '[\\O-1]', '[\\O]', '[\\S]',
# in ZINC
'[PH1+1]', '[S@@+1]', '[=S@]', '[\\S@]', '[S+1]', '[=SH1+1]', '[\\N+1]', '[/O+1]', 
'[/C@@]', '[-\\Ring1]', '[/NH1-1]', '[\\NH1]', '[-/Ring2]', '[/NH1]', '[\\NH2+1]', 
'[/C@]', '[=O+1]', '[=P]', '[PH1]', '[\\S-1]', '[\\I]', '[/C@H1]', '[=PH2]', 
'[/C@@H1]', '[/F]', '[/N+1]', '[\\Br]', '[\\N-1]', '[\\Cl]', '[\\NH1+1]', '[\\F]', 
'[=P@@]', '[P@@H1]', '[P+1]', '[=S+1]', '[\\C@H1]', '[=N-1]', '[CH2-1]', '[-/Ring1]', 
'[CH1-1]', '[/NH2+1]', '[\\C@@H1]', '[/S@]', '[=OH1+1]', '[=P@]', '[/Br]', '[/N-1]', 
'[P@]', '[#N+1]']
TOKENS_DEEPSMILES = TOKENS + ['%20', '%22', '%28', '%12', 
'%15', '%13', '%10', '%16', '%11', 
'%14', '%21', '%18', '%17', '%19']

TOKEN2ATOMFEAT = {
    "[B]": (5, 0, 0),
    "[B-]": (5, -1, 0),
    "[BH2-]": (5, -1, 2),
    "[BH3-]": (5, -1, 3),
    "[BH-]": (5, -1, 1),
    "[CH]": (6, 0, 1),
    "[CH2]": (6, 0, 2),
    "[CH-]": (6, -1, 1),
    "[CH2-]": (6, -1, 2),
    "[C]": (6, 0, 0),
    "[C+]": (6, 1, 0),
    "[CH+]": (6, 1, 1),
    "[CH2+]": (6, 1, 2),
    "[C-]": (6, -1, 0),
    "[N-]": (7, -1, 0),
    "[NH-]": (7, -1, 1),
    "[N]": (7, 0, 0),
    "[NH]": (7, 0, 1),
    "[N+]": (7, 1, 0),
    "[NH+]": (7, 1, 1),
    "[NH2+]": (7, 1, 2),
    "[NH3+]": (7, 1, 3),
    "[O-]": (8, -1, 0),
    "[O]": (8, 0, 0),
    "[O+]": (8, 1, 0),
    "[OH+]": (8, 1, 1),
    "[F]": (9, 0, 0),
    "[F+]": (9, 1, 0),
    "[F-]": (9, -1, 0),
    "[Si]": (14, 0, 0),
    "[Si-]": (14, -1, 0),
    "[SiH-]": (14, -1, 1),
    "[SiH2]": (14, 0, 2),
    "[Si+]": (14, 0, 1),
    "[P]": (15, 0, 0),
    "[PH]": (15, 0, 1),
    "[PH2]": (15, 0, 2),
    "[P+]": (15, 1, 0),
    "[PH+]": (15, 1, 1),
    "[PH2+]": (15, 1, 2),
    "[P-]": (15, -1, 0),
    "[S-]": (16, -1, 0),
    "[SH-]": (16, -1, 1),
    "[S]": (16, 0, 0),
    "[S+]": (16, 1, 0),
    "[SH]": (16, 0, 1),
    "[SH+]": (16, 1, 1),
    "[Cl]": (17, 0, 0),
    "[Cl+]": (17, 1, 0),
    "[Cl-]": (17, -1, 0),
    "[Cl++]": (17, 2, 0),
    "[Cl+++]": (17, 3, 0),
    "[Se]": (34, 0, 0),
    "[Se+]": (34, 1, 0),
    "[Se-]": (34, -1, 0),
    "[SeH]": (34, 0, 1),
    "[SeH2]": (34, 0, 2),
    "[Br]": (35, 0, 0),
    "[Br++]": (35, 2, 0),
    "[Br-]": (35, -1, 0),
    "[I]": (53, 0, 0),
    "[I+]": (53, 1, 0),
    "[IH2]": (53, 0, 2),
    "[I++]": (53, 2, 0),
    "[IH]": (53, 0, 1),
    "[I+++]": (53, 3, 0),
    "[I-]": (53, -1, 0)
}
ATOMFEAT2TOKEN = {val: key for key, val in TOKEN2ATOMFEAT.items()}

MAX_LEN = 250

@enum.unique
class TokenType(enum.IntEnum):
    ATOM = 1
    BOND = 2
    BRANCH_START = 3
    BRANCH_END = 4
    RING_NUM = 5
    SPECIAL = 6


ORGANIC_ATOMS = "B C N O P S F Cl Br I * b c n o s p".split()
BOND_TYPE_DICT = {1: 5, 2: 6, 3: 7, 1.5: 8}
NODE_TYPE_DICT = {key: idx+9 for idx, key in enumerate(TOKEN2ATOMFEAT.keys())}
VALENCE_DICT = {'B': 3, 'C': 4, 'N': 3, 'O': 2, 'P': 3, 'S': 2, 'F': 1, 
                'Cl': 1, 'Br': 1, 'I': 1, 'b': 3, 'c': 4, 'n': 3, 
                'o': 2, 's': 2, 'p': 3,
                # ions
                "[C@@H]": 4, "[C@@]": 4, "[C@H]": 4, "[C@]": 4,
                "[CH-]": 2, "[CH2-]": 1, "[H]": 1, "[N+]": 4, "[N-]": 2, "[NH+]": 3, "[NH-]": 1, "[NH2+]": 2, "[NH3+]": 1, 
                "[O+]": 3, "[O-]": 1, "[OH+]": 2, "[P+]": 4, "[P@@H]": 2, "[P@@]": 3, "[P@]": 3, "[PH+]": 3, 
                "[PH2]": 1, "[PH]": 2, "[S+]": 3, "[S-]": 1, "[S@@+]": 3, "[S@@]": 2, 
                "[S@]": 2, "[SH+]": 2, "[n+]": 4, "[n-]": 2, "[nH+]": 3, "[nH]": 2, 
                "[o+]": 3, "[s+]": 3, "[C-]": 3, "[c-]": 3, "[cH-]": 2}

def token_to_id(tokens, num_additional_tokens=0):
    # num_additional_tokens: to add other CoT tokens or normal tokens (e.g., multiset tokens need to be placed after node tokens)
    return {token: tokens.index(token)+num_additional_tokens for token in tokens}

def id_to_token(tokens):
    return {idx: tokens[idx] for idx in range(len(tokens))}
    
def map_length(token):
    
    if token in [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, PASS_TOKEN]:
        return 0
    
    if token[0] == '[':
        return 0
    elif token[0] == '▁':
        return len(token)-2
    else:
        return len(token)-1

def tokenize_selfies(selfies):
    
    selfies_split = selfies.split("]")[:-1]
    tokens = ["[bos]"]
    tokens.extend([selfies + "]" for selfies in selfies_split])
    tokens.append("[eos]")
    TOKEN2ID = token_to_id(TOKENS_SELFIES)
    return [TOKEN2ID[token] for token in tokens]

def tokenize(smiles, string_type='smiles'):
    org_smiles = smiles
    mol = Chem.MolFromSmiles(org_smiles)
    if mol is None:
        return []
    atoms = iter(mol.GetAtoms())
    smiles = iter(smiles)
    tokens = ["[bos]"]
    peek = None
    valence_token = None
    ring_token = None
    implicit_h_token = None
    while True:
        char = peek if peek else next(smiles, "")
        peek = None
        if not char:
            break

        if char == "[":
            token = char
            for char in smiles:
                token += char
                if char == "]":
                    break
            # add ion
            if string_type in ['smiles_val_ion', 'smiles_val_ion_ring', 'smiles_val_ion_ring_imph']:
                valence_token = '[' + str(VALENCE_DICT[token]) + ']'
            # ring
            if string_type in ['smiles_val_ion_ring', 'smiles_val_ion_ring_imph']:
                clean_atom = token.replace('[', '').replace(']', '').replace('+', '').replace('-', '').replace('H', '').replace('+', '').replace('3', '').replace('2', '').replace('@', '')
                atom = next(atoms, "")
                atom_symbol = atom.GetSymbol()
                atom_is_ring = atom.IsInRing()
                atom_implicit_h = atom.GetNumImplicitHs()
                if atom_symbol.lower() == clean_atom.lower():
                    ring_token = '[R]' if atom_is_ring else '[NR]'
                else:
                    raise ValueError(f"Atom symbol {atom_symbol} does not match token {clean_atom}")
                # implicit H
                if string_type == 'smiles_val_ion_ring_imph':
                    implicit_h_token = '[H' + str(atom_implicit_h) + ']'

        elif char in ORGANIC_ATOMS:
            peek = next(smiles, "")
            if char + peek in ORGANIC_ATOMS:
                token = char + peek
                peek = None
            else:
                token = char

            
            if string_type in ['smiles_val', 'smiles_val_ion', 'smiles_val_ion_ring', 'smiles_val_ion_ring_imph']:
                valence_token = '[' + str(VALENCE_DICT[token]) + ']'
            if string_type in ['smiles_val_ion_ring', 'smiles_val_ion_ring_imph']:
                atom = next(atoms, "")
                atom_symbol = atom.GetSymbol()
                atom_is_ring = atom.IsInRing()
                atom_implicit_h = atom.GetNumImplicitHs()
                # ring
                if atom_symbol.lower() == token.lower():
                    ring_token = '[R]' if atom_is_ring else '[NR]'
                else:
                    raise ValueError(f"Atom symbol {atom_symbol} does not match token {token}")
                # implicit H
                if string_type == 'smiles_val_ion_ring_imph':
                    implicit_h_token = '[H' + str(atom_implicit_h) + ']'


        elif char == "%":
            token = char + next(smiles, "") + next(smiles, "")
            print(token)
            print(TOKEN2ID[token])
            assert False

        elif char in "-=#$:.()%/\\" or char.isdigit():
            token = char
        else:
            raise ValueError(f"Undefined tokenization for chararacter {char}")

        tokens.append(token)
        if valence_token:
            tokens.append(valence_token)
            valence_token = None
        if ring_token:
            tokens.append(ring_token)
            ring_token = None
        if implicit_h_token:
            tokens.append(implicit_h_token)
            implicit_h_token = None
        
        
    tokens.append("[eos]")
    # TOKEN2ID = token_to_id(TOKENS)
    # result_token_ids = [TOKEN2ID.get(token, "") for token in tokens]
    # return [token for token in result_token_ids if token != ""]
    return tokens

def untokenize(sequence, string_type='smiles'):
    
    if string_type in ['smiles', 'smiles_val', 'smiles_val_ion', 'smiles_val_ion_ring', 'smiles_val_ion_ring_imph']:
        ID2TOKEN = id_to_token(TOKENS)
    else:
        raise ValueError(f"Undefined string type {string_type}")

    # sequence = [s for s in sequence if s in TOKENS]
    tokens = [ID2TOKEN[id_] for id_ in sequence if id_ in ID2TOKEN.keys()]
    if BOS_TOKEN not in tokens:
        return ""
    tokens = tokens[tokens.index(BOS_TOKEN):]
    if len(tokens) == 0:
        return ""
    if tokens[0] != "[bos]":
        return ""
    elif "[eos]" not in tokens:
        return ""

    tokens = tokens[1 : tokens.index("[eos]")]
    return "".join(tokens)

def get_char_tokens(string_type):
    tokens = {
        'smiles': TOKENS,
        'selfies': TOKENS_SELFIES,
        'deep_smiles': TOKENS_DEEPSMILES
    }.get(string_type)
    return tokens


def map_tokens(string_type):
    tokens = get_char_tokens(string_type)
    
    return tokens