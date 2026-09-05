#!/usr/bin/env python3
"""
find_correct_af_model.py — Find the AlphaFold model matching expected sequence length.

Usage: python3 src/fetch/find_correct_af_model.py <uniprot_id> <expected_length>

Queries the AlphaFold prediction endpoint, finds the model whose length
(uniprotEnd - uniprotStart + 1) matches expected_length, and prints the
entryId (e.g. AF-O43236-7-F1). Prints nothing if no match is found.
"""
import sys
import json
import urllib.request
import urllib.error


def find_model(uniprot_id: str, expected_length: int) -> str | None:
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        req = urllib.request.urlopen(url, timeout=15)
        data = json.loads(req.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None

    for model in data:
        model_length = model['uniprotEnd'] - model['uniprotStart'] + 1
        if model_length == expected_length:
            return model['entryId']
    return None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <uniprot_id> <expected_length>", file=sys.stderr)
        sys.exit(1)

    uid = sys.argv[1]
    try:
        length = int(sys.argv[2])
    except ValueError:
        print(f"Error: expected_length must be an integer, got '{sys.argv[2]}'", file=sys.stderr)
        sys.exit(1)

    result = find_model(uid, length)
    if result:
        print(result)
