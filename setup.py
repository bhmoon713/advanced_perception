from setuptools import setup

package_name = 'advanced_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'yolo_object_detection = advanced_perception.yolo_object_detection:main',
        'yolo_segmentation = advanced_perception.yolo_segmentation:main',
        'circle_detection = advanced_perception.circle_detection:main',
        'fruit_mask_saver = advanced_perception.fruit_mask_saver:main',
        'inverse_transform_node = advanced_perception.inverse_transform_node:main',
        'transform_test = advanced_perception.transform_test:main',
        'tilt_corrected_circle_detector = advanced_perception.tilt_corrected_circle_detector:main',
        'topdown_circle_detection = advanced_perception.topdown_circle_detection:main',
        ],
    },
)
