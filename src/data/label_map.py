"""
6-class label taxonomy: Benign, Reconnaissance, DoS_DDoS, Injection_Exploit,
BruteForce, Botnet_C2.  One dict per dataset.  Values of None are dropped.
"""

LYCOS_IDS2017 = {
    "benign":                   "Benign",
    "portscan":                 "Reconnaissance",
    "ddos":                     "DoS_DDoS",
    "dos_goldeneye":            "DoS_DDoS",
    "dos_hulk":                 "DoS_DDoS",
    "dos_slowhttptest":         "DoS_DDoS",
    "dos_slowloris":            "DoS_DDoS",
    "ftp_patator":              "BruteForce",
    "ssh_patator":              "BruteForce",
    "webattack_bruteforce":     "BruteForce",
    "webattack_sql_injection":  "Injection_Exploit",
    "webattack_xss":            "Injection_Exploit",
    "heartbleed":               "Injection_Exploit",
    "bot":                      "Botnet_C2",
}

CIC_IDS2018 = {
    "Benign":                   "Benign",
    "Bot":                      "Botnet_C2",
    "Brute Force -Web":         "BruteForce",
    "Brute Force -XSS":         "Injection_Exploit",
    "DDOS attack-HOIC":         "DoS_DDoS",
    "DDOS attack-LOIC-UDP":     "DoS_DDoS",
    "DDoS attacks-LOIC-HTTP":   "DoS_DDoS",
    "DoS attacks-GoldenEye":    "DoS_DDoS",
    "DoS attacks-Hulk":         "DoS_DDoS",
    "DoS attacks-SlowHTTPTest": "DoS_DDoS",
    "DoS attacks-Slowloris":    "DoS_DDoS",
    "FTP-BruteForce":           "BruteForce",
    "SSH-Bruteforce":           "BruteForce",
    "Infilteration":            "Injection_Exploit",
    "SQL Injection":            "Injection_Exploit",
}

UNSW_NB15 = {
    "Normal":        "Benign",
    "":              "Benign",
    "Reconnaissance":"Reconnaissance",
    "Analysis":      "Reconnaissance",
    "DoS":           "DoS_DDoS",
    "Generic":       "DoS_DDoS",
    "Fuzzers":       "Injection_Exploit",
    "Exploits":      "Injection_Exploit",
    "Shellcode":     "Injection_Exploit",
    "Backdoors":     "Botnet_C2",
    "Worms":         "Botnet_C2",
}

TON_IOT = {
    "Benign":    "Benign",
    "scanning":  "Reconnaissance",
    "ddos":      "DoS_DDoS",
    "dos":       "DoS_DDoS",
    "injection": "Injection_Exploit",
    "xss":       "Injection_Exploit",
    "mitm":      "Injection_Exploit",
    "password":  "BruteForce",
    "ransomware":"Botnet_C2",
    "Backdoor":  "Botnet_C2",
}

LABEL_MAPS = {
    "lycos_ids2017": LYCOS_IDS2017,
    "cic_ids2018":   CIC_IDS2018,
    "unsw_nb15":     UNSW_NB15,
    "ton_iot":       TON_IOT,
}

CLASSES = ["Benign", "Reconnaissance", "DoS_DDoS", "Injection_Exploit", "BruteForce", "Botnet_C2"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
