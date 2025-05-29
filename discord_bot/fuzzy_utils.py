from fuzzywuzzy import fuzz

def find_fuzzy_match(term, text, threshold=85):
    words = text.split()
    candidates = [" ".join(words[i:i+len(term.split())]) for i in range(len(words))]
    for segment in candidates:
        if fuzz.partial_ratio(term.lower(), segment.lower()) >= threshold:
            return segment
    return None

def find_all_fuzzy_terms(terms_list, text, threshold=85, exclude=None):
    words = text.split()
    matched_terms = []
    for term in terms_list:
        if exclude and term == exclude:
            continue
        candidates = [" ".join(words[i:i+len(term.split())]) for i in range(len(words))]
        for segment in candidates:
            if fuzz.partial_ratio(term.lower(), segment.lower()) >= threshold:
                matched_terms.append(term)
                break
    return matched_terms
