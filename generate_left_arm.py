"""
generate_left_arm.py  —  mirror SO-101 right arm → left arm XML
─────────────────────────────────────────────────────────────────
Prefixes every name with "l_" to avoid MuJoCo name collisions,
repositions the root body to the left-arm base, and mirrors it
so it faces inward (+X direction, toward the right arm).

Arms are 12 inches (0.305 m) apart, centred on X=0:
  Right arm base:  x= 0.152  (set inside so101_new_calib.xml)
  Left  arm base:  x=-0.152  (set here via LEFT_BASE_POS)

Usage:
    python3 generate_left_arm.py \\
        --input  so101_new_calib.xml \\
        --output so101_left_calib.xml
"""

import argparse
import xml.etree.ElementTree as ET

NAME_VALUE_ATTRS = {
    "body1","body2","joint1","joint2","site1","site2",
    "joint","site","tendon","body","objname","anchor",
}
NAME_DEF_ATTRS = {"name"}
PREFIX = "l_"

# ── Left arm base position ────────────────────────────────────────────────
# 12 inches (0.3048 m) apart → each arm is 0.152 m from X=0
LEFT_BASE_POS = "-0.1016 0.20 0.02"   # was "-0.15 0.64 0.02"
LEFT_BASE_EULER = "0 0 180"            # mirror to face inward (+X direction)


def prefix_value(val):
    return " ".join(PREFIX + t for t in val.split())


def process_element(el, is_root_body=False):
    for attr, val in list(el.attrib.items()):
        if attr in NAME_DEF_ATTRS:
            el.set(attr, PREFIX + val)
        elif attr in NAME_VALUE_ATTRS:
            el.set(attr, prefix_value(val))
        # class= is intentionally NOT prefixed — shared with right arm defaults

    if is_root_body and el.tag.lower() == "body":
        el.set("pos", LEFT_BASE_POS)
        # Remove any existing rotation attrs, then set our mirror rotation
        for a in ("euler", "quat", "axisangle", "xyaxes", "zaxis"):
            el.attrib.pop(a, None)
        el.set("euler", LEFT_BASE_EULER)

    for child in el:
        process_element(child)


parser = argparse.ArgumentParser()
parser.add_argument("--input",  required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

tree = ET.parse(args.input)
root = tree.getroot()

# Strip sections that must not be duplicated in the parent scene
for tag in ("asset", "compiler", "default"):
    for el in root.findall(tag):
        root.remove(el)
        print(f"Removed <{tag}>")

worldbody = root.find("worldbody")
done = False
if worldbody is not None:
    for child in worldbody:
        is_root = child.tag.lower() == "body" and not done
        process_element(child, is_root_body=is_root)
        if is_root:
            done = True

for section in ("actuator", "sensor", "tendon", "contact", "equality"):
    sec = root.find(section)
    if sec is not None:
        for child in sec:
            process_element(child)

tree.write(args.output, encoding="unicode", xml_declaration=True)
print(f"Done → {args.output}")
print(f"Left arm base: {LEFT_BASE_POS}  euler: {LEFT_BASE_EULER}")