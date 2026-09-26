# field normalisation: transliteration, accents, abbreviations, tokens
import re, polars as pl
from indic_transliteration import sanscript as S
from indic_transliteration.sanscript import transliterate
SCRIPTS = [((0x0900, 0x097F), S.DEVANAGARI), ((0x0980, 0x09FF), S.BENGALI), ((0x0A00, 0x0A7F), S.GURMUKHI), ((0x0A80, 0x0AFF), S.GUJARATI),
           ((0x0B00, 0x0B7F), S.ORIYA), ((0x0B80, 0x0BFF), S.TAMIL), ((0x0C00, 0x0C7F), S.TELUGU), ((0x0C80, 0x0CFF), S.KANNADA), ((0x0D00, 0x0D7F), S.MALAYALAM)]
NONLATIN = r"[\x{0900}-\x{0DFF}]"
def _translit(s):
    out = []
    for w in s.split(" "):
        sc = None
        for ch in w:
            o = ord(ch)
            for (a, b), name in SCRIPTS:
                if a <= o <= b: sc = name; break
            if sc: break
        out.append(transliterate(w, sc, S.ITRANS) if sc else w)
    t = " ".join(out).lower()
    # squash ITRANS phonetics toward english spelling
    for a, b in (("aa", "a"), ("ii", "i"), ("uu", "u"), ("~n", "n"), ("sh", "s"), ("ph", "f"), ("chh", "ch"), ("\\.", ""), ("^", ""), ("m", "m"), ("r^i", "ri")):
        t = t.replace(a, b)
    return t
def translit_col(col):
    # only strings containing indic chars go through python
    return pl.when(pl.col(col).str.contains(NONLATIN)).then(pl.col(col).map_elements(_translit, return_dtype=pl.String)).otherwise(pl.col(col))

LEGAL = "llc|l l c|inc|incorporated|corp|corporation|co|company|ltd|limited|pvt|private|llp|pllc|lp|plc|pc|sa|sas|sasu|sarl|eurl|sci|snc|gmbh|the|and|of|des|de|du|la|le|les|et|pvt ltd|opc|dba|doing business as|prāiveta|praivet|limited|limiteda|limitad|praa"
GENERIC = "group|services|service|center|centre|enterprises|enterprise|solutions|holdings|associates|industries|international|global|india|consultants|co|company|trading|traders|ventures|partners|agency|management|systems|technologies|technology"
STATES = {"alabama":"al","alaska":"ak","arizona":"az","arkansas":"ar","california":"ca","colorado":"co","connecticut":"ct","delaware":"de","florida":"fl","georgia":"ga","hawaii":"hi","idaho":"id","illinois":"il","indiana":"in","iowa":"ia","kansas":"ks","kentucky":"ky","louisiana":"la","maine":"me","maryland":"md","massachusetts":"ma","michigan":"mi","minnesota":"mn","mississippi":"ms","missouri":"mo","montana":"mt","nebraska":"ne","nevada":"nv","new hampshire":"nh","new jersey":"nj","new mexico":"nm","new york":"ny","north carolina":"nc","north dakota":"nd","ohio":"oh","oklahoma":"ok","oregon":"or","pennsylvania":"pa","rhode island":"ri","south carolina":"sc","south dakota":"sd","tennessee":"tn","texas":"tx","utah":"ut","vermont":"vt","virginia":"va","washington":"wa","west virginia":"wv","wisconsin":"wi","wyoming":"wy","district of columbia":"dc"}
ABBR = {"street":"st","road":"rd","avenue":"ave","av":"ave","drive":"dr","lane":"ln","boulevard":"blvd","bd":"blvd","bld":"blvd","court":"ct","place":"pl","highway":"hwy","parkway":"pkwy","circle":"cir","terrace":"ter","square":"sq","trail":"trl","way":"way","north":"n","south":"s","east":"e","west":"w","apartment":"apt","suite":"ste","unit":"unit","rue":"r","allee":"all","chemin":"ch","impasse":"imp","route":"rte","place":"pl","saint":"st","sainte":"ste","nagar":"ngr","colony":"col","sector":"sec","floor":"flr","building":"bldg","near":"nr","opposite":"opp","opp":"opp","house":"h","number":"no","plot":"plot","door":"no","hno":"h no","flat":"flat","mount":"mt","fort":"ft"}
ADDR_STOP = {"no", "h", "n/a", "na", "unit", "apt", "ste", "flr", "nr", "near", "po", "p", "o", "dist", "pmb", "india", "usa", "france", "#"}

def base(col):
    # lowercase, strip accents, punctuation -> space
    return (pl.col(col).str.to_lowercase().str.normalize("NFKD").str.replace_all(r"\p{M}", "")
            .str.replace_all(r"&", " and ").str.replace_all(r"[^\p{L}\p{N} ]", " ").str.replace_all(r"\s+", " ").str.strip_chars())

def add_norm(df):
    df = df.with_columns(translit_col("business_name").alias("_n"), translit_col("business_address").alias("_a"))
    df = df.with_columns(base("_n").alias("nn"), base("_a").alias("an"),
                         pl.col("business_name").str.contains(NONLATIN).alias("nonlatin"),
                         (pl.col("business_address") == "").alias("addr_empty"),
                         pl.col("business_name").str.contains(r"(?i)\.(com|in|net|org|co|fr|biz|info)\b").alias("is_domain"))
    df = df.with_columns(pl.col("nn").str.replace_all(r"\b(com|net|org|in|fr|biz|info|www)\b", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("nn"))
    # address canonical: states -> abbr, street types -> abbr, strip leading zeros
    a = pl.col("an")
    for k, v in STATES.items(): a = a.str.replace_all(rf"\b{k}\b", v)
    a = a.str.split(" ").list.eval(pl.element().replace(ABBR)).list.join(" ")
    a = a.str.replace_all(r"\b0+(\d)", "$1")
    df = df.with_columns(a.alias("an"))
    df = df.with_columns(
        pl.col("nn").str.replace_all(rf"\b({LEGAL})\b", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("nc"),
        pl.col("an").str.extract_all(r"\d+").list.unique().alias("nums"),
    )
    df = df.with_columns(pl.col("nc").str.replace_all(" ", "").alias("nsp"),
                         pl.col("nc").str.replace_all(rf"\b({GENERIC})\b", " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("ncc"))
    df = df.with_columns(pl.col("nc").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("init"))
    return df.drop("_n", "_a")
