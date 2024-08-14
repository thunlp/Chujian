from argparse import ArgumentParser
from pathlib import Path
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple


def dict_size(d) -> int:
    return sum(len(v) for v in d.values())


def dump_json(data, file):
    json.dump(
        data,
        open(file, 'w', encoding='utf8'),
        ensure_ascii=False,
        indent=4,
    )


def merged_glyphs(glyphs: List[str]) -> defaultdict[str, list]:
    '''
    Return {new_name: [old_names]}
    '''
    # Map new glyph name to old glyph name
    new_to_old_name = defaultdict(list)
    # Discard all glyphs containing these chars (after preprocessing label)
    DISCARD_CHARS = [
        '?'
        '□', '■',
        '○', '●',
        '△', '▲',
        '☆', '★',
        '◇', '◆',
        '□'
    ]

    for glyph in glyphs:
        orig = glyph
        # Normalize the glyph label
        RM_STRS = [
            '=', 'None'
        ]
        for c in RM_STRS:
            glyph = glyph.replace(c, '')

        # Replace brackets
        for c in ['（', '〈', '[']:
            glyph = glyph.replace(c, '(')
        for c in ['）', '〉', ']']:
            glyph = glyph.replace(c, ')')

        if glyph == '':
            continue

        if glyph[-1] == ')':
            for i in range(len(glyph) - 2, -1, -1):
                if glyph[i] == '(':
                    # "（*）"
                    if glyph[i] == '(':
                        if glyph[i+1:-1] == '○':
                            glyph = glyph[:i]
                        else:
                            glyph = glyph[i+1:-1]
                    else:
                        # "*}（*）"
                        if glyph[i-1] == '}':
                            glyph = glyph[i+1:-1]
                        # "A（*）" -> "A"
                        else:
                            glyph = glyph[0]
                    break
            else:
                glyph = glyph[:-1]
        # "A→B"
        if '→' in glyph:
            glyph = glyph.split('→')[1]
        if glyph == '𬨭':
            glyph = '將'
        if glyph == '𫵖':
            glyph = '尸示'

        if any(c in glyph for c in DISCARD_CHARS):
            # if '○' in glyph:
            #     print(orig)
            #     print(glyph)
            #     exit()
            continue
        new_to_old_name[glyph].append(orig)
    return new_to_old_name


def get_glyph_to_files(src_dir: Path) -> Dict[str, List[str]]:
    glyph_to_files = {}
    for src_glyph_dir in src_dir.iterdir():
        if not src_glyph_dir.is_dir():
            continue
        name = src_glyph_dir.name
        glyph_to_files[name] = []
        for file in src_glyph_dir.iterdir():
            glyph_to_files[name].append(str(file))
    return glyph_to_files


def split_data(glyph_to_files: dict) -> Tuple[dict, dict, dict]:
    # Split into train and test set
    # For each class, split by 8:1:1 ratio.
    print("Splitting...")
    train_images = {}
    dev_images = {}
    test_images = {}
    for glyph, image_files in glyph_to_files.items():
        random.seed(0)
        random.shuffle(image_files)
        # Floored to make sure test and dev has at least one example.
        split_idx = [
            int(len(image_files) * 0.9),
            int(len(image_files) * 0.95),
        ]
        train_images[glyph] = image_files[: split_idx[0]]
        dev_images[glyph] = image_files[split_idx[0]:split_idx[1]]
        test_images[glyph] = image_files[split_idx[1]:]
    return train_images, dev_images, test_images


def split_and_dump(glyph_to_files: dict, dst_dir: Path):
    train_images, dev_images, test_images = split_data(glyph_to_files)

    print("Train images:", dict_size(train_images))
    print("Dev images:", dict_size(dev_images))
    print("Test images:", dict_size(test_images))

    dump_json(train_images, dst_dir / "train.json")
    dump_json(dev_images, dst_dir / "dev.json")
    dump_json(test_images, dst_dir / "test.json")


def merge_and_dump(src_dir: Path, dst_dir: Path, k: int):
    print('Getting glyph to files...')
    orig_glyph_to_files = get_glyph_to_files(src_dir)
    print('Merging glyphs...')
    print(f'Before: {len(orig_glyph_to_files)} glyphs')
    glyphs = list(orig_glyph_to_files.keys())
    new_to_old_name = merged_glyphs(glyphs)
    print(f'After: {len(new_to_old_name)} glyphs')

    # Building new glyph to files
    glyph_to_files = {}
    for new_name, old_names in new_to_old_name.items():
        glyph_to_files[new_name] = []
        for old_name in old_names:
            glyph_to_files[new_name].extend(orig_glyph_to_files[old_name])

    num_examples = sum(len(files) for files in glyph_to_files.values())
    print(f'Found {num_examples} examples')

    # Remove the glyphs with less than k samples
    print(f'Removing glyphs with less than {k} samples...')
    glyph_to_files = {
        glyph: files
        for glyph, files in glyph_to_files.items() if len(files) >= k}
    print(f'After: {len(glyph_to_files)} glyphs')
    num_examples = sum(len(files) for files in glyph_to_files.values())
    print(f'Found {num_examples} examples')

    glyph_to_cnt = {k: len(v) for k, v in glyph_to_files.items()}

    # Sort by descending count
    merged_sorted = sorted(
        glyph_to_cnt.items(), key=lambda x: x[1], reverse=True)
    glyph_to_cnt = {k: v for k, v in merged_sorted}

    glyph_to_cnt_file = dst_dir / 'glyph_to_count_sorted.json'
    print(f'Dumping to {glyph_to_cnt_file}')
    dump_json(glyph_to_cnt, glyph_to_cnt_file)

    merged_to_orig_file = dst_dir / 'new_to_orig_name.json'
    print(f'Dumping to {merged_to_orig_file}')
    dump_json(new_to_old_name, merged_to_orig_file)

    glyph_to_files_file = dst_dir / "glyph_to_files.json"
    print(f'Dumping to {glyph_to_files_file}')
    dump_json(glyph_to_files, glyph_to_files_file)

    # Split and dump
    split_and_dump(glyph_to_files, dst_dir)


def main():
    p = ArgumentParser()
    p.add_argument('-k', type=int, default=0)
    p.add_argument('--src_dir', type=str, default='../data')
    p.add_argument('--dst_dir', type=str, default='./data')
    args = p.parse_args()
    src_dir = args.src_dir / 'glyphs'
    print(f'========= k = {args.k} =========')
    dst_dir = args.dst_dir / f'glyphs_k-{args.k}'
    dst_dir.mkdir(exist_ok=True, parents=True)
    merge_and_dump(src_dir, dst_dir, args.k)


if __name__ == '__main__':
    main()
