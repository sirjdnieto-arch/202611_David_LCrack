# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared configuration — tickers, names, groups.
# Imported by both app.py and indicators.py to avoid circular imports.
# ─────────────────────────────────────────────────────────────────────────────

ALL_TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","GOOG","META","BRK-B","TSLA","JNJ","V",
    "PG","XOM","UNH","JPM","HD","LLY","MA","CVX","ABBV","KO","PEP",
    "COST","BAC","CRM","NFLX","ABT","MCD","LMT","EL","NEE","CAT","MRK",
    "TPL","ASML","ADBE","AVGO","CSCO","CMCSA","AMD","TXN","QCOM","AMAT","LITE","LRCX","COHR","CMI",
    "NEM","ULTA","IT","FOXA","LUV","VLO","ADP","FN","POET","KEYS","HPE","MRVL","BRKR","AAOI",
    "INTU","VRTX","ZS","PLTR","CSU.TO","MU","LVMUY","SAP","OR.PA","TTE","ON","MELI","CTSH","THRY","QBTS","RGTI","IONQ",
    "MC.PA","SIE.DE","ENGI.PA","AIR.PA","ALV.DE","EL.PA","AI.PA","BNP.PA",
    "SAN.PA","KER.PA","SU.PA","NESN.SW","LIN.DE","VOW3.DE","BMW.DE","ADS.DE",
    "IFX.DE","MUV2.DE","FRE.DE","DTE.DE","RWE.DE","ITX.MC","BBVA.MC","SAN.MC",
    "TEF.MC","IBE.MC","REP.MC","FER.MC","ACX.MC","ACS.MC","AENA.MC","ANA.MC",
    "IAG.MC","LOG.MC","MAP.MC","PUIG.MC","NTGY.MC","ELE.MC","IDR.MC","PDD",
    "NIO","TCEHY","BZUN","FUTU","MOMO","MNSO","TAL","EDU","WB","XPEV",
    "GC=F","SI=F","BTC-USD","ETH-USD","XRP-USD","SOL-USD","CRCL","PYPL",
    "PGR","LULU","GDDY","ZTS","SAP.DE","IT","ADSK","BKNG","GEN","EXPE","RMD","CPAY","EFX"
]

TICKER_NAMES = {
    "AAPL":"Apple","MSFT":"Microsoft","AMZN":"Amazon","NVDA":"NVIDIA","GOOG":"Alphabet",
    "META":"Meta","BRK-B":"Berkshire","TSLA":"Tesla","JNJ":"Johnson&Johnson","V":"Visa",
    "PG":"Procter&Gamble","XOM":"ExxonMobil","UNH":"UnitedHealth","JPM":"JPMorgan",
    "HD":"Home Depot","LLY":"Eli Lilly","MA":"Mastercard","CVX":"Chevron","ABBV":"AbbVie",
    "KO":"Coca-Cola","PEP":"PepsiCo","COST":"Costco","BAC":"Bank of America","CRM":"Salesforce",
    "NFLX":"Netflix","ABT":"Abbott","MCD":"McDonald's","LMT":"Lockheed Martin","EL":"Estée Lauder",
    "NEE":"NextEra","CAT":"Caterpillar","MRK":"Merck","TPL":"Texas Pacific","ASML":"ASML",
    "ADBE":"Adobe","AVGO":"Broadcom","CSCO":"Cisco","CMCSA":"Comcast","AMD":"AMD",
    "TXN":"Texas Instr.","QCOM":"Qualcomm","AMAT":"Applied Materials","LITE":"Lumentum",
    "LRCX":"Lam Research","COHR":"Coherent","CMI":"Cummins","NEM":"Newmont","ULTA":"Ulta Beauty",
    "IT":"Gartner","FOXA":"Fox Corp","LUV":"Southwest","VLO":"Valero","ADP":"ADP","FN":"Fabrinet",
    "POET":"POET Technologies","KEYS":"Keysight","HPE":"HPE","MRVL":"Marvell","BRKR":"Bruker",
    "AAOI":"Applied Opt.","INTU":"Intuit","VRTX":"Vertex","ZS":"Zscaler","PLTR":"Palantir",
    "CSU.TO":"Constellation Soft.","MU":"Micron","LVMUY":"LVMH","SAP":"SAP","OR.PA":"L'Oréal",
    "TTE":"TotalEnergies","ON":"ON Semi","MELI":"MercadoLibre",
    "CTSH":"Cognizant","THRY":"Thryv","KLTR":"Kaltura","QBTS":"D-Wave","RGTI":"Rigetti",
    "IONQ":"IonQ","MC.PA":"LVMH (Paris)","SIE.DE":"Siemens","ENGI.PA":"Engie",
    "AIR.PA":"Airbus","ALV.DE":"Allianz","EL.PA":"EssilorLuxottica","AI.PA":"Air Liquide",
    "BNP.PA":"BNP Paribas","SAN.PA":"Sanofi","KER.PA":"Kering","SU.PA":"Schneider",
    "NESN.SW":"Nestlé","LIN.DE":"Linde","VOW3.DE":"Volkswagen","BMW.DE":"BMW",
    "ADS.DE":"Adidas","IFX.DE":"Infineon","MUV2.DE":"Munich Re","FRE.DE":"Fresenius",
    "DTE.DE":"Deutsche Telekom","RWE.DE":"RWE","ITX.MC":"Inditex","BBVA.MC":"BBVA",
    "SAN.MC":"Santander","TEF.MC":"Telefónica","IBE.MC":"Iberdrola","REP.MC":"Repsol",
    "FER.MC":"Ferrovial","ACX.MC":"Acerinox","ACS.MC":"ACS","AENA.MC":"Aena",
    "ANA.MC":"Acciona","IAG.MC":"IAG","LOG.MC":"Logista","MAP.MC":"Mapfre",
    "PUIG.MC":"Puig","NTGY.MC":"Naturgy","ELE.MC":"Endesa","IDR.MC":"Indra",
    "PDD":"PDD Holdings","NIO":"NIO","TCEHY":"Tencent","BZUN":"Baozun","FUTU":"Futu",
    "MOMO":"Hello Group","MNSO":"Miniso","TAL":"TAL Education","EDU":"New Oriental",
    "WB":"Weibo","XPEV":"XPeng","GC=F":"Oro","SI=F":"Plata","BTC-USD":"Bitcoin",
    "ETH-USD":"Ethereum","XRP-USD":"Ripple","SOL-USD":"Solana","CRCL":"Circle","PYPL":"Paypal",
    "PGR":"Progressive Corporation","LULU":"Lululemon","GDDY":"GoDaddy","ZTS":"Zoetis","SAP.DE":"SAP",
    "IT":"Gartner","ADSK":"Autodesk","BKNG":"Booking","GEN":"Gen Digital","EXPE":"Expedia Group","RMD":"ResMed",
    "CPAY":"Corpay","EFX":"Equifax"
}

GRUPOS = {
    "Todos":             ALL_TICKERS,
    "US Large Cap":      ["AAPL","MSFT","AMZN","NVDA","GOOG","META","BRK-B","TSLA",
                          "JNJ","V","PG","XOM","UNH","JPM","HD","LLY","MA","CVX","ABBV",
                          "KO","PEP","COST","BAC","CRM","NFLX","ABT","MCD","LMT","EL",
                          "NEE","CAT","MRK"],
    "Tecnología":        ["AAPL","MSFT","NVDA","GOOG","META","TSLA","ADBE","AVGO","CSCO",
                          "AMD","TXN","QCOM","AMAT","LRCX","INTU","VRTX","ZS","PLTR","MU",
                          "LITE","ON","ASML","SAP","SIE.DE","IFX.DE","AI.PA"],
    "Europa":            ["MC.PA","SIE.DE","ENGI.PA","AIR.PA","ALV.DE","EL.PA","AI.PA",
                          "BNP.PA","SAN.PA","KER.PA","SU.PA","NESN.SW","LIN.DE","VOW3.DE",
                          "BMW.DE","ADS.DE","IFX.DE","MUV2.DE","FRE.DE","DTE.DE","RWE.DE",
                          "OR.PA","TTE"],
    "España":            ["ITX.MC","BBVA.MC","SAN.MC","TEF.MC","IBE.MC","REP.MC","FER.MC",
                          "ACX.MC","ACS.MC","AENA.MC","ANA.MC","IAG.MC","LOG.MC","MAP.MC",
                          "PUIG.MC","NTGY.MC","ELE.MC","IDR.MC"],
    "China / Asia":      ["PDD","NIO","TCEHY","BZUN","FUTU","MOMO","MNSO","TAL","EDU","WB","XPEV"],
    "Crypto / Materias": ["GC=F","SI=F","BTC-USD","ETH-USD","XRP-USD"],
}
