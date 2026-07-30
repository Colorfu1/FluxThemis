import hashlib
import re
from pathlib import Path
from xml.etree import ElementTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MESSAGE_PACKAGE = PROJECT_ROOT / 'ros' / 'fluxthemis_msgs'
ROS2_MESSAGE_PACKAGE = PROJECT_ROOT / 'ros2' / 'fluxthemis_msgs'
NOETIC_MANIFEST = PROJECT_ROOT / 'ros' / 'noetic_py310.rosinstall'


def test_predict_action_service_contract_is_stable():
    service = MESSAGE_PACKAGE / 'srv' / 'PredictAction.srv'

    assert service.read_text(encoding='utf-8').splitlines() == [
        'std_msgs/Header header',
        'string request_id',
        'string episode_id',
        'bool reset',
        'int64 seed',
        'string prompt',
        'string[] image_names',
        'sensor_msgs/Image[] images',
        'string[] state_names',
        'uint32[] state_sizes',
        'float32[] state_values',
        'string unnorm_key',
        '---',
        'std_msgs/Header header',
        'string request_id',
        'bool ok',
        'string error',
        'float32[] actions',
        'uint32 action_horizon',
        'uint32 action_dim',
        'bool denormalized',
        'float64 inference_time_s',
    ]


def test_report_evaluation_service_contract_is_stable():
    service = MESSAGE_PACKAGE / 'srv' / 'ReportEvaluation.srv'

    assert service.read_text(encoding='utf-8').splitlines() == [
        'std_msgs/Header header',
        'string request_id',
        'string run_id',
        'uint64 sequence',
        'string event_type',
        'string payload_json',
        '---',
        'std_msgs/Header header',
        'string request_id',
        'bool accepted',
        'string error',
        'string run_dir',
    ]


def test_ros_package_declares_message_dependencies():
    package_root = ElementTree.parse(MESSAGE_PACKAGE / 'package.xml').getroot()
    dependencies = {(element.tag, element.text)
                    for element in package_root
                    if element.tag.endswith('depend')}

    assert ('buildtool_depend', 'catkin') in dependencies
    assert ('build_depend', 'message_generation') in dependencies
    assert ('exec_depend', 'message_runtime') in dependencies
    assert ('depend', 'sensor_msgs') in dependencies
    assert ('depend', 'std_msgs') in dependencies


def test_ros2_service_contract_matches_ros1_exactly():
    for service_name in ('PredictAction.srv', 'ReportEvaluation.srv'):
        ros1_service = MESSAGE_PACKAGE / 'srv' / service_name
        ros2_service = ROS2_MESSAGE_PACKAGE / 'srv' / service_name

        assert ros2_service.read_bytes() == ros1_service.read_bytes()


def test_ros2_package_declares_rosidl_dependencies():
    package_root = ElementTree.parse(ROS2_MESSAGE_PACKAGE /
                                     'package.xml').getroot()
    dependencies = {(element.tag, element.text)
                    for element in package_root
                    if element.tag.endswith('depend')}
    groups = {(element.tag, element.text)
              for element in package_root if element.tag == 'member_of_group'}

    assert ('buildtool_depend', 'ament_cmake') in dependencies
    assert ('buildtool_depend', 'rosidl_default_generators') in dependencies
    assert ('exec_depend', 'rosidl_default_runtime') in dependencies
    assert ('depend', 'sensor_msgs') in dependencies
    assert ('depend', 'std_msgs') in dependencies
    assert ('member_of_group', 'rosidl_interface_packages') in groups

    cmake = (ROS2_MESSAGE_PACKAGE /
             'CMakeLists.txt').read_text(encoding='utf-8')
    assert 'rosidl_generate_interfaces' in cmake
    assert 'ament_package()' in cmake
    assert package_root.find('./export/build_type').text == 'ament_cmake'


def test_noetic_manifest_checksum_matches_installer_and_sdist():
    installer = (PROJECT_ROOT / 'scripts' /
                 'install_ros1_noetic_py310.sh').read_text(encoding='utf-8')
    match = re.search(
        r'^SOURCE_MANIFEST_SHA256="([0-9a-f]{64})"$',
        installer,
        flags=re.MULTILINE)
    assert match is not None
    assert hashlib.sha256(NOETIC_MANIFEST.read_bytes()).hexdigest() == \
        match.group(1)

    manifest_in = (PROJECT_ROOT / 'MANIFEST.in').read_text(encoding='utf-8')
    assert 'include ros/noetic_py310.rosinstall' in manifest_in.splitlines()
