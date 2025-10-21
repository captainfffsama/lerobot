#!/usr/bin/env python3
"""
Test script for Meta Quest 3 teleoperator integration with lerobot.
This script demonstrates how to use the MetaQuest3Teleop class.
"""

import logging
import time
from config_meta_quest3 import MetaQuest3Config
from teleop_meta_quest3 import MetaQuest3Teleop

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_meta_quest3_teleop():
    """Test the Meta Quest 3 teleoperator."""
    
    # Create configuration
    config = MetaQuest3Config(
        id="meta_quest3_test",
        ipaddress="localhost",
        port="30001"
    )
    
    # Create teleoperator
    teleop = MetaQuest3Teleop(config)
    
    try:
        # Test connection
        logger.info("Testing Meta Quest 3 teleoperator...")
        logger.info(f"Action features: {teleop.action_features}")
        logger.info(f"Feedback features: {teleop.feedback_features}")
        
        # Connect
        logger.info("Connecting to Meta Quest 3...")
        teleop.connect(calibrate=False)  # Skip calibration for testing
        
        # Test connection status
        logger.info(f"Connected: {teleop.is_connected}")
        logger.info(f"Calibrated: {teleop.is_calibrated}")
        
        # Test getting actions (this will work even without calibration)
        logger.info("Testing action retrieval...")
        for i in range(5):
            try:
                action = teleop.get_action()
                if action:
                    logger.info(f"Action {i+1}: {list(action.keys())}")
                    # Log some key values
                    if 'left_hand.pos' in action:
                        logger.info(f"  Left hand position: {action['left_hand.pos']}")
                    if 'right_hand.pos' in action:
                        logger.info(f"  Right hand position: {action['right_hand.pos']}")
                    if 'enabled' in action:
                        logger.info(f"  Teleoperation enabled: {action['enabled']}")
                else:
                    logger.info(f"Action {i+1}: No data available")
            except Exception as e:
                logger.warning(f"Error getting action {i+1}: {e}")
            
            time.sleep(1.0)
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    
    finally:
        # Disconnect
        try:
            teleop.disconnect()
            logger.info("Disconnected from Meta Quest 3")
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")


def test_configuration():
    """Test the configuration class."""
    logger.info("Testing MetaQuest3Config...")
    
    config = MetaQuest3Config(
        id="test_config",
        ipaddress="192.168.1.100",
        port="30001"
    )
    
    logger.info(f"Config ID: {config.id}")
    logger.info(f"Config IP: {config.ipaddress}")
    logger.info(f"Config Port: {config.port}")
    logger.info("Configuration test passed!")


if __name__ == "__main__":
    print("=== Meta Quest 3 Teleoperator Test ===")
    
    # Test configuration
    test_configuration()
    print()
    
    # Test teleoperator (this will fail if no Meta Quest 3 is connected)
    print("Note: This test requires a Meta Quest 3 device connected to the network.")
    print("The test will timeout if no device is found.")
    print()
    
    try:
        test_meta_quest3_teleop()
    except Exception as e:
        print(f"Test failed (expected if no Meta Quest 3 is connected): {e}")
        print("This is normal if no Meta Quest 3 device is available.")
