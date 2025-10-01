#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_msgs.msg import Yolov8Segmentation
import numpy as np
import cv2
import os
from advanced_perception.yolo_segmentation import YoloSegmentation 
import ultralytics

bridge = CvBridge()

class FruitMaskSaver(Node):
    def __init__(self) -> None:
        """Initializes the FruitMaskSaver node."""
        super().__init__('fruit_mask_saver')
        self.subscription = self.create_subscription(
            Yolov8Segmentation,
            '/Yolov8_segmentation',
            self.handle_segmentation,
            10)
        self.img_sub = self.create_subscription(
            Image,
            '/deepmind_robot1/deepmind_robot1_camera/image_raw',
            self.image_callback,
            10)
        
        self.output_dir = '/home/user/ros2_ws/src/advanced_perception/data/segmentation_results'
        os.makedirs(self.output_dir, exist_ok=True)
        self.img = None
        self.yolov8_segmentation = []
        self.image_ready = False
        self.segmentation_ready = False

        # Create an instance of YoloSegmentation
        self.yolo_segmentation_instance = YoloSegmentation()

    def handle_segmentation(self, msg: Yolov8Segmentation) -> None:
        """ Handles incoming segmentation data """
        self.yolov8_segmentation = msg.yolov8_segmentation
        self.segmentation_ready = True

    def image_callback(self, data: Image) -> None:
        """ Callback function for image data received from the camera """
        self.img = bridge.imgmsg_to_cv2(data, "bgr8")
        self.image_ready = True

    def save_masks_as_image(self, results: ultralytics.engine.results, output_dir: str) -> None:
        """ Saves the masks as images in the specified directory """
        if not hasattr(results, 'masks') or results.masks is None:
            self.get_logger().error("No 'masks' attribute found in the results or masks are empty. Please check the model configuration.")
            return

        masks = results.masks.data.cpu().numpy()
        combined_mask = self.combine_masks(masks, results)
        self.save_combined_mask(combined_mask, output_dir)

    def combine_masks(self, masks: np.ndarray, results: ultralytics.engine.results) -> np.ndarray:
        """ Combines individual masks into a single mask image """
        combined_mask = np.zeros((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)
        apple_count = 0
        banana_count = 0

        for i, mask in enumerate(masks):
            class_id = int(results.boxes[i].cls)
            class_name = self.yolo_segmentation_instance.model.names[class_id]

            if class_name not in ["apple", "banana"]:
                continue

            mask_data = mask
            if mask_data.size != 0 and mask_data.ndim >= 2:
                if mask_data.shape[0] == 1:
                    mask_data = np.squeeze(mask_data, axis=0)
                mask_data = (mask_data * 255).astype(np.uint8)
                mask_data = cv2.resize(mask_data, (self.img.shape[1], self.img.shape[0]))
                
                color = [0, 255, 0] if class_name == "apple" else [0, 255, 255]
                combined_mask[mask_data > 0] = color

                M = cv2.moments(mask_data)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    count = apple_count + 1 if class_name == 'apple' else banana_count + 1
                    cv2.putText(combined_mask, f"{class_name[0].upper()}{count}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if class_name == "apple":
                    apple_count += 1
                elif class_name == "banana":
                    banana_count += 1

        self.get_logger().info(f"Total apples: {apple_count}, Total bananas: {banana_count}")
        return combined_mask

    def save_combined_mask(self, combined_mask: np.ndarray, output_dir: str) -> None:
        """ Saves the combined mask image to the specified directory """
        mask_path = os.path.join(output_dir, "combined_masks.png")
        cv2.imwrite(mask_path, combined_mask)
        self.get_logger().info(f"Successfully created and saved the mask image inside {self.output_dir} folder")

def main(args=None) -> None:
    rclpy.init(args=args)
    fruit_mask_saver = FruitMaskSaver()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(fruit_mask_saver)

    try:
        while rclpy.ok():
            executor.spin_once()
            if fruit_mask_saver.image_ready and fruit_mask_saver.segmentation_ready:
                if fruit_mask_saver.img is not None and fruit_mask_saver.yolov8_segmentation:
                    results = fruit_mask_saver.yolo_segmentation_instance.model(fruit_mask_saver.img)
                    if results:
                        fruit_mask_saver.save_masks_as_image(results[0], fruit_mask_saver.output_dir)
                    break  # Process only once
            else:
                if fruit_mask_saver.segmentation_ready is False:
                    fruit_mask_saver.get_logger().error("Waiting to subscribe to the Segmentation results ..")
                else:
                    fruit_mask_saver.get_logger().error("Waiting to subscribe to the Image data ..")
    finally:
        fruit_mask_saver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()