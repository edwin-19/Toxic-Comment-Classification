bag_of_words = [
    ['\n', " "],
    ['\?', " "], 
    ["\[.*\]"," "],
    ["\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"," "],
    ["don't","do not"],
    ["doesn't", "does not"],
    ["didn't", "did not"],
    ["hasn't", "has not"],
    ["haven't", "have not"],
    ["hadn't", "had not"],
    ["won't", "will not"],
    ["wouldn't", "would not"],
    ["can't", "can not"],
    ["cannot", "can not"],
    ["i'm", "i am"],
    ["i'll", "i will"],
    ["its", "it is"],
    ["it's", "it is"],
    ["that's", "that is"],
    ["weren't", "were not"],
    ["i'd","i would"],
    ["i've","i have"],
    ["she'd","she would"],
    ["they'll","they will"],
    ["they're","they are"],
    ["we'd","we would"],
    ["we'll","we will"],
    ["we've","we have"],
    ["it'll","it will"],
    ["there's","there is"],
    ["where's","where is"],
    ["they're","they are"],
    ["let's","let us"],
    ["couldn't","could not"],
    ["shouldn't","should not"],
    ["wasn't","was not"],
    ["could've","could have"],
    ["might've","might have"],
    ["must've","must have"],
    ["should've","should have"],
    ["would've","would have"],
    ["who's","who is"],
    ["\bim\b", "i am"],
    [r'[^\w\s]',''],
    ["\d+", ""],
    ["what's", "what is "],
    [r" e g ", " eg "],
    [r" b g ", " bg "],
    [r" 9 11 ", "911"],
    
    [r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}", ""],
    [r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", ""],
    
    # Drop image
    [r"image:[a-zA-Z0-9]*\.jpg", " ",],
    [r"image:[a-zA-Z0-9]*\.png", " "],
    [r"image:[a-zA-Z0-9]*\.gif", " "],
    [r"image:[a-zA-Z0-9]*\.bmp", " "],
    
    # Drop css
    [r"#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})", " "],
    [r"\{\|[^\}]*\|\}", " "],
    
    # Clean templates
    [r"\[?\[user:.*\|", " "],
    [r"\[?\[wikipedia:.*\]", " "],
    [r"\[?\[special:.*\]", " "],
    [r"\[?\[category:.*\]", " "]
]