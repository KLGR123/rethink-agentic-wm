#!/usr/bin/env python3
"""Write the Lucene JsonCollection input for WebShop's search index.
"""

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WebShop Lucene documents.jsonl from a product file")
    parser.add_argument("--webshop-path", default="/fs_mol/liujiarun/WebShop")
    parser.add_argument("--file-path", default="/fs_mol/liujiarun/WebShop/data/items_shuffle.json")
    parser.add_argument("--out", default="/fs_mol/liujiarun/WebShop/search_engine/resources/documents.jsonl")
    args = parser.parse_args()

    if args.webshop_path not in sys.path:
        sys.path.insert(0, args.webshop_path)
    from web_agent_site.engine.engine import load_products # noqa: PLC0415

    print(f"Loading products from {args.file_path} (this parses around 5.5 GB) ...")
    all_products, *_ = load_products(filepath=args.file_path, num_products=None, human_goals=True)
    print(f"Loaded {len(all_products)} products. Writing docs ...")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w+") as f:
        for p in all_products:
            options = p.get("options", {})
            option_texts = []
            for option_name, option_contents in options.items():
                option_contents_text = ", ".join(option_contents)
                option_texts.append(f"{option_name}: {option_contents_text}")
            option_text = ", and ".join(option_texts)

            doc = {
                "id": p["asin"],
                "contents": " ".join([
                    p["Title"],
                    p["Description"],
                    p["BulletPoints"][0],
                    option_text,
                ]).lower(),
                "product": p,
            }
            f.write(json.dumps(doc) + "\n")

    print(f"Wrote {len(all_products)} docs to {args.out}")


if __name__ == "__main__":
    main()
