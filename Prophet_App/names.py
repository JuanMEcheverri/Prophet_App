"""Nombre de empresa por ticker (Nasdaq-100), para mostrar junto al ticker en app y artifact."""
NAMES = {
    "AAPL": "Apple", "ABNB": "Airbnb", "ADBE": "Adobe", "ADI": "Analog Devices", "AEP": "American Electric Power",
    "AMAT": "Applied Materials", "AMD": "Advanced Micro Devices", "AMGN": "Amgen", "AMZN": "Amazon", "ANSS": "Ansys",
    "APP": "AppLovin", "ASML": "ASML Holding", "AVGO": "Broadcom", "BIIB": "Biogen", "BKNG": "Booking Holdings",
    "BMRN": "BioMarin", "CDNS": "Cadence Design Systems", "CEG": "Constellation Energy", "COST": "Costco", "CPRT": "Copart",
    "CRWD": "CrowdStrike", "CSCO": "Cisco", "CTAS": "Cintas", "DDOG": "Datadog", "DLTR": "Dollar Tree", "DXCM": "Dexcom",
    "EA": "Electronic Arts", "EBAY": "eBay", "ENPH": "Enphase Energy", "EXC": "Exelon", "FAST": "Fastenal",
    "FSLR": "First Solar", "FTNT": "Fortinet", "GEHC": "GE HealthCare", "GFS": "GlobalFoundries", "GILD": "Gilead Sciences",
    "GOOG": "Alphabet (clase C)", "GOOGL": "Alphabet (clase A)", "HON": "Honeywell", "IDXX": "IDEXX Laboratories",
    "ILMN": "Illumina", "INTU": "Intuit", "ISRG": "Intuitive Surgical", "KHC": "Kraft Heinz", "KLAC": "KLA",
    "LRCX": "Lam Research", "MAR": "Marriott", "MDB": "MongoDB", "MDLZ": "Mondelez", "MELI": "MercadoLibre",
    "META": "Meta Platforms", "MNST": "Monster Beverage", "MRVL": "Marvell Technology", "MSFT": "Microsoft",
    "MU": "Micron Technology", "NFLX": "Netflix", "NTAP": "NetApp", "NVDA": "NVIDIA", "NXPI": "NXP Semiconductors",
    "ODFL": "Old Dominion Freight Line", "OKTA": "Okta", "ON": "ON Semiconductor", "ORLY": "O'Reilly Automotive",
    "PANW": "Palo Alto Networks", "PAYX": "Paychex", "PCAR": "PACCAR", "PDD": "PDD Holdings (Temu)", "PEP": "PepsiCo",
    "PYPL": "PayPal", "QCOM": "Qualcomm", "REGN": "Regeneron", "ROP": "Roper Technologies", "ROST": "Ross Stores",
    "SBUX": "Starbucks", "SMCI": "Super Micro Computer", "SNPS": "Synopsys", "SWKS": "Skyworks", "TEAM": "Atlassian",
    "TMUS": "T-Mobile US", "TSLA": "Tesla", "TTD": "The Trade Desk", "TXN": "Texas Instruments", "VRSK": "Verisk Analytics",
    "VRTX": "Vertex Pharmaceuticals", "WBD": "Warner Bros. Discovery", "WDAY": "Workday", "XEL": "Xcel Energy",
    "ZBRA": "Zebra Technologies", "ZM": "Zoom", "ZS": "Zscaler", "QQQ": "Invesco QQQ (Nasdaq-100)",
}


def label(ticker):
    n = NAMES.get(ticker)
    return f"{ticker} · {n}" if n else ticker
