"""Export the local training robot and PD parameters for Isaac Lab visualization."""

import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET

import mujoco
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML, G1_ACTION_SCALE
from mjlab.entity import Entity
from g1_supermimic.config import resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="assets/isaac_g1", type=resolve_path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(G1_XML)
    compiler = tree.getroot().find("compiler")
    mesh_dir = (G1_XML.parent / compiler.get("meshdir", "")).resolve()
    shutil.copytree(mesh_dir, output / "meshes", dirs_exist_ok=True)
    compiler.set("meshdir", "meshes")
    # Isaac's importer does not resolve this asset's nested default classes.
    # Expand them without changing the source training asset.
    classes = {"": {}}

    def collect(node, inherited):
        attributes = {tag: values.copy() for tag, values in inherited.items()}
        for child in node:
            if child.tag != "default":
                attributes.setdefault(child.tag, {}).update(child.attrib)
        classes[node.get("class", "")] = attributes
        for child in node.findall("default"):
            collect(child, attributes)

    for default in tree.getroot().findall("default"):
        collect(default, {})

    def expand(node, inherited=""):
        child_class = node.attrib.pop("childclass", inherited)
        cls = node.attrib.pop("class", inherited)
        attributes = classes.get(cls, {}).get(node.tag, {}).copy()
        attributes.update(node.attrib)
        node.attrib.clear()
        node.attrib.update(attributes)
        if node.tag == "joint":
            node.set("type", "hinge")
        for child in node:
            expand(child, child_class)

    expand(tree.getroot().find("worldbody"))
    for default in tree.getroot().findall("default"):
        tree.getroot().remove(default)
    tree.write(output / "g1.xml")
    original = mujoco.MjModel.from_xml_path(str(G1_XML))
    expanded = mujoco.MjModel.from_xml_path(str(output / "g1.xml"))
    import numpy as np

    for field in (
        "body_mass",
        "body_pos",
        "body_quat",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "jnt_axis",
        "jnt_range",
    ):
        np.testing.assert_allclose(
            getattr(original, field), getattr(expanded, field), atol=1e-8
        )
    cfg = get_g1_robot_cfg()
    model = Entity(cfg).spec.compile()
    joint_ids = [
        j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    names = [model.joint(j).name for j in joint_ids]
    actuator = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu)}
    values = {
        k: {}
        for k in (
            "stiffness",
            "damping",
            "effort",
            "armature",
            "default_q",
            "action_scale",
            "target_min",
            "target_max",
        )
    }
    for j, name in zip(joint_ids, names):
        a = actuator[j]
        values["stiffness"][name] = float(model.actuator_gainprm[a, 0])
        values["damping"][name] = float(-model.actuator_biasprm[a, 2])
        values["effort"][name] = float(model.actuator_forcerange[a, 1])
        values["armature"][name] = float(model.dof_armature[model.jnt_dofadr[j]])
        values["target_min"][name], values["target_max"][name] = map(
            float, model.actuator_ctrlrange[a]
        )
        for key, table in (
            ("default_q", cfg.init_state.joint_pos),
            ("action_scale", G1_ACTION_SCALE),
        ):
            matches = [v for pattern, v in table.items() if re.fullmatch(pattern, name)]
            if key == "action_scale" and len(matches) != 1:
                raise ValueError(f"Ambiguous action scale for {name}")
            values[key][name] = float(matches[-1] if matches else 0)
    (output / "control.json").write_text(
        json.dumps(
            {
                "joint_names": names,
                "schema_version": 1,
                "mjcf": "g1.xml",
                "control_dt": 0.02,
                "physics_dt": 0.005,
                **values,
            },
            indent=2,
        )
        + "\n"
    )
    print(output / "control.json")


if __name__ == "__main__":
    main()
