import sys
import os
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add megabite_core to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'megabite_core'))

from megabite_core.matrix_bot import RibitMatrixBot

async def run_mock_test():
    """
    Runs the Matrix Bot in mock mode to test command handling.
    """
    logger.info("Starting Megabite Matrix Bot in Mock Test Mode...")
    
    # Use dummy credentials for mock mode
    homeserver = "https://mock.homeserver.net"
    username = "@mock_user:mock.homeserver.net"
    password = "mock_password"
    
    # The bot will automatically fall back to mock mode if matrix-nio is not available
    # or if it fails to connect, but we will explicitly call the mock run function
    # to test the command logic.
    
    bot = RibitMatrixBot(homeserver, username, password)
    
    # Since the bot's start method handles the mock mode fallback, we call start.
    # We pass a dummy client object to the word learner's learn_for_duration method
    # to prevent errors during the mock run.
    
    # Note: The actual _run_mock_mode contains an infinite loop with input().
    # For a non-interactive test, we'll simulate the command execution directly.
    
    # --- Simulated Command Test ---
    mock_sender = "@rabit233:matrix.anarchists.space" # Authorized user
    mock_room = "!mock_room:mock.homeserver.net"
    
    print("\n--- Testing ?thought_experiment command ---")
    response_te = await bot._process_message("ribit.2.0: ?thought_experiment The nature of consciousness", mock_sender, mock_room)
    print(f"Response: {response_te}")
    
    print("\n--- Testing ?opinion command ---")
    response_op = await bot._process_message("ribit.2.0: ?opinion The future of AI ethics", mock_sender, mock_room)
    print(f"Response: {response_op}")
    
    print("\n--- Testing ?learn_words command (will fail gracefully in mock mode) ---")
    response_lw = await bot._process_message("ribit.2.0: ?learn_words 1", mock_sender, mock_room)
    print(f"Response: {response_lw}")
    
    print("\n--- Testing ?help command ---")
    response_help = await bot._process_message("ribit.2.0: ?help", mock_sender, mock_room)
    print(f"Response: {response_help}")
    
    print("\n--- Mock Test Complete ---")


if __name__ == '__main__':
    try:
        asyncio.run(run_mock_test())
    except KeyboardInterrupt:
        print("\nMock Test interrupted.")
    except Exception as e:
        logger.error(f"An error occurred during mock test: {e}")
        import traceback
        traceback.print_exc()
