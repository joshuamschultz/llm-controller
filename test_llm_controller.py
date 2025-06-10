# test_llm_controller.py
"""
Comprehensive test suite for LLMController class
Includes unit tests, integration tests, and behavioral tests

Run with:
    pytest test_llm_controller.py -v
    or
    python -m pytest test_llm_controller.py -v --tb=short
"""

import pytest
import os
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import asyncio

# Import the LLMController (adjust import based on your file structure)
try:
    from llm_controller import LLMController, GrokChatModel, HuggingFaceChatModel
except ImportError:
    # If the class is in the same file, we'll define it here for testing
    # In practice, you'd have this in a separate module
    print("Warning: Could not import LLMController. Make sure it's available.")

from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# Test configuration
TEST_CONFIG = {
    "mock_mode": not bool(os.getenv("ANTHROPIC_API_KEY")),  # Use mocks if no API key
    "run_integration_tests": bool(os.getenv("ANTHROPIC_API_KEY")),
    "test_timeout": 30,  # seconds
}

class TestLLMControllerUnit:
    """Unit tests for LLMController basic functionality"""
    
    def test_initialization_valid_provider(self):
        """Test controller initializes with valid provider"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_claude.return_value = Mock()
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            
            assert controller.llm_name == "claude-3-sonnet-20240229"
            assert controller.provider == "claude"
            assert controller._current_model is not None
    
    def test_initialization_invalid_provider(self):
        """Test controller raises error with invalid provider"""
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMController(llm="test-model", provider="invalid_provider")
    
    def test_switch_model_same_provider(self):
        """Test switching models within same provider"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_claude.return_value = Mock()
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            
            # Switch to different model
            controller.switch_model(llm="claude-3-haiku-20240307")
            
            assert controller.llm_name == "claude-3-haiku-20240307"
            assert controller.provider == "claude"
    
    def test_switch_model_different_provider(self):
        """Test switching to different provider"""
        with patch('llm_controller.ChatAnthropic') as mock_claude, \
             patch('llm_controller.ChatOpenAI') as mock_openai:
            
            mock_claude.return_value = Mock()
            mock_openai.return_value = Mock()
            
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            controller.switch_model(llm="gpt-4", provider="openai")
            
            assert controller.llm_name == "gpt-4"
            assert controller.provider == "openai"
    
    def test_current_model_info(self):
        """Test current_model_info property"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_model = Mock()
            mock_model.__class__.__name__ = "ChatAnthropic"
            mock_claude.return_value = mock_model
            
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            info = controller.current_model_info
            
            assert info["provider"] == "claude"
            assert info["model"] == "claude-3-sonnet-20240229"
            assert info["type"] == "ChatAnthropic"
    
    def test_llm_type(self):
        """Test _llm_type method"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_claude.return_value = Mock()
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            
            assert controller._llm_type() == "llm_controller_claude"


class TestLLMControllerMocked:
    """Mocked behavioral tests that don't require API keys"""
    
    def test_invoke_delegation(self):
        """Test that invoke is properly delegated to underlying model"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.content = "Test response"
            mock_model.invoke.return_value = mock_response
            mock_claude.return_value = mock_model
            
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            response = controller.invoke("Test prompt")
            
            mock_model.invoke.assert_called_once_with("Test prompt", None)
            assert response.content == "Test response"
    
    def test_stream_delegation(self):
        """Test that stream is properly delegated"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_model = Mock()
            mock_stream = [Mock(content="chunk1"), Mock(content="chunk2")]
            mock_model.stream.return_value = iter(mock_stream)
            mock_claude.return_value = mock_model
            
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            chunks = list(controller.stream("Test prompt"))
            
            mock_model.stream.assert_called_once()
            assert len(chunks) == 2
            assert chunks[0].content == "chunk1"
    
    def test_generate_delegation(self):
        """Test that _generate is properly delegated"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_model = Mock()
            mock_result = Mock()
            mock_model._generate.return_value = mock_result
            mock_claude.return_value = mock_model
            
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            messages = [HumanMessage(content="Test")]
            result = controller._generate(messages)
            
            mock_model._generate.assert_called_once_with(messages, None, None)
            assert result == mock_result
    
    def test_attribute_delegation(self):
        """Test that unknown attributes are delegated to underlying model"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_model = Mock()
            mock_model.some_custom_method.return_value = "custom_result"
            mock_claude.return_value = mock_model
            
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            result = controller.some_custom_method()
            
            assert result == "custom_result"
    
    def test_missing_attribute_error(self):
        """Test that missing attributes raise AttributeError"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_model = Mock()
            del mock_model.nonexistent_method  # Ensure it doesn't exist
            mock_claude.return_value = mock_model
            
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            
            with pytest.raises(AttributeError):
                controller.nonexistent_method()


class TestLLMControllerChains:
    """Test LangChain integration with mocked models"""
    
    def test_prompt_chain_integration(self):
        """Test controller works in LangChain chains"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.content = "Chain response"
            mock_model.invoke.return_value = mock_response
            mock_claude.return_value = mock_model
            
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            
            # Create a simple chain
            prompt = ChatPromptTemplate.from_template("Explain {topic}")
            chain = prompt | controller | StrOutputParser()
            
            result = chain.invoke({"topic": "testing"})
            assert result == "Chain response"
    
    def test_message_based_invocation(self):
        """Test controller works with message-based input"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.content = "Message response"
            mock_model.invoke.return_value = mock_response
            mock_claude.return_value = mock_model
            
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            messages = [HumanMessage(content="Hello")]
            
            response = controller.invoke(messages)
            assert response.content == "Message response"


@pytest.mark.skipif(not TEST_CONFIG["run_integration_tests"], 
                   reason="No API key available for integration tests")
class TestLLMControllerIntegration:
    """Integration tests that require actual API keys"""
    
    @pytest.fixture
    def controller(self):
        """Fixture providing a real LLMController instance"""
        return LLMController(llm="claude-3-haiku-20240307", provider="claude")
    
    def test_real_invoke(self, controller):
        """Test real API call with invoke"""
        response = controller.invoke("Say 'Hello World' and nothing else.")
        assert response.content is not None
        assert len(response.content) > 0
        assert "Hello World" in response.content or "hello world" in response.content.lower()
    
    def test_real_message_conversation(self, controller):
        """Test real conversation with messages"""
        messages = [
            HumanMessage(content="My name is TestUser."),
            AIMessage(content="Hello TestUser! Nice to meet you."),
            HumanMessage(content="What's my name?")
        ]
        
        response = controller.invoke(messages)
        assert "TestUser" in response.content
    
    def test_real_streaming(self, controller):
        """Test real streaming functionality"""
        chunks = []
        for chunk in controller.stream("Count from 1 to 3, one number per word."):
            chunks.append(chunk.content)
        
        full_response = "".join(chunks)
        assert len(chunks) > 1  # Should have multiple chunks
        assert any(char.isdigit() for char in full_response)  # Should contain numbers
    
    def test_real_model_switching(self, controller):
        """Test switching between real models"""
        # Test with Haiku
        response1 = controller.invoke("Say 'Haiku' and nothing else.")
        assert "Haiku" in response1.content or "haiku" in response1.content.lower()
        
        # Switch to Sonnet
        controller.switch_model(llm="claude-3-sonnet-20240229", provider="claude")
        response2 = controller.invoke("Say 'Sonnet' and nothing else.")
        assert "Sonnet" in response2.content or "sonnet" in response2.content.lower()
    
    @pytest.mark.timeout(TEST_CONFIG["test_timeout"])
    def test_response_time_reasonable(self, controller):
        """Test that responses come back in reasonable time"""
        start_time = time.time()
        controller.invoke("Say hello.")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 30  # Should respond within 30 seconds


class TestLLMControllerBehavioral:
    """Behavioral tests for real-world usage patterns"""
    
    def test_error_handling_invalid_model_switch(self):
        """Test graceful error handling with invalid model"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_claude.side_effect = Exception("Invalid model")
            
            with pytest.raises(Exception):
                LLMController(llm="invalid-claude-model", provider="claude")
    
    def test_multiple_rapid_switches(self):
        """Test rapid model switching doesn't break controller"""
        with patch('llm_controller.ChatAnthropic') as mock_claude, \
             patch('llm_controller.ChatOpenAI') as mock_openai:
            
            mock_claude.return_value = Mock()
            mock_openai.return_value = Mock()
            
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            
            # Rapid switches
            for i in range(5):
                if i % 2 == 0:
                    controller.switch_model(llm="gpt-4", provider="openai")
                else:
                    controller.switch_model(llm="claude-3-sonnet-20240229", provider="claude")
            
            # Should end up with last configuration
            assert controller.provider == "claude"
    
    def test_concurrent_usage_thread_safety(self):
        """Test that controller handles concurrent access gracefully"""
        import threading
        import queue
        
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.content = "Thread response"
            mock_model.invoke.return_value = mock_response
            mock_claude.return_value = mock_model
            
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            results = queue.Queue()
            
            def worker():
                try:
                    response = controller.invoke("Test from thread")
                    results.put(("success", response.content))
                except Exception as e:
                    results.put(("error", str(e)))
            
            # Start multiple threads
            threads = [threading.Thread(target=worker) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # Check results
            success_count = 0
            while not results.empty():
                status, result = results.get()
                if status == "success":
                    success_count += 1
                    assert result == "Thread response"
            
            assert success_count == 3


class TestPerformance:
    """Performance and load testing"""
    
    def test_initialization_performance(self):
        """Test that controller initializes quickly"""
        with patch('llm_controller.ChatAnthropic') as mock_claude:
            mock_claude.return_value = Mock()
            
            start_time = time.time()
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            end_time = time.time()
            
            init_time = end_time - start_time
            assert init_time < 1.0  # Should initialize in under 1 second
    
    def test_switch_performance(self):
        """Test that model switching is fast"""
        with patch('llm_controller.ChatAnthropic') as mock_claude, \
             patch('llm_controller.ChatOpenAI') as mock_openai:
            
            mock_claude.return_value = Mock()
            mock_openai.return_value = Mock()
            
            controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
            
            start_time = time.time()
            controller.switch_model(llm="gpt-4", provider="openai")
            end_time = time.time()
            
            switch_time = end_time - start_time
            assert switch_time < 0.5  # Should switch in under 0.5 seconds


# Test utilities and fixtures
@pytest.fixture
def mock_environment():
    """Fixture that sets up mock environment variables"""
    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test_key_123',
        'OPENAI_API_KEY': 'test_openai_key',
    }):
        yield


@pytest.fixture
def sample_messages():
    """Fixture providing sample messages for testing"""
    return [
        HumanMessage(content="Hello, I'm testing the system."),
        AIMessage(content="Hello! I'm ready to help you test."),
        HumanMessage(content="Can you repeat my first message?")
    ]


# Test configuration and markers
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test requiring API keys"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Main test runner
if __name__ == "__main__":
    print("LLMController Test Suite")
    print("=" * 50)
    print(f"Mock mode: {TEST_CONFIG['mock_mode']}")
    print(f"Integration tests: {TEST_CONFIG['run_integration_tests']}")
    print()
    
    # Run tests programmatically
    import subprocess
    import sys
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Exit code: {result.returncode}")
        
    except Exception as e:
        print(f"Error running tests: {e}")
        print("Try running manually with: pytest test_llm_controller.py -v")