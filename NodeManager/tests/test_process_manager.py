"""
Tests for ProcessManager with mock Petals processes.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch

from kwaainet_node.core.process_manager import ProcessManager
from kwaainet_node.core.models import ProcessState, ResourceLimits
from kwaainet_node.platform.base import ProcessController


class MockProcessHandle:
    """Mock process handle for testing"""

    def __init__(self, pid: int = 12345):
        self.pid = pid
        self.platform_handle = Mock()


class MockProcessController(ProcessController):
    """Mock process controller for testing"""

    def __init__(self):
        self.processes = {}
        self.next_pid = 10000

    async def spawn_process(self, cmd, env, cwd=None):
        pid = self.next_pid
        self.next_pid += 1
        handle = MockProcessHandle(pid)
        self.processes[pid] = {
            'handle': handle,
            'cmd': cmd,
            'env': env,
            'running': True
        }
        return handle

    async def kill_process(self, handle, signal_num=None):
        if handle.pid in self.processes:
            self.processes[handle.pid]['running'] = False
        return True

    async def get_process_info(self, handle):
        from kwaainet_node.core.models import ProcessInfo
        return ProcessInfo(
            pid=handle.pid,
            memory_rss=1024*1024*100,  # 100MB
            memory_vms=1024*1024*200,  # 200MB
            cpu_percent=15.5,
            status="running",
            create_time=1234567890.0
        )

    async def is_process_running(self, handle):
        return self.processes.get(handle.pid, {}).get('running', False)


@pytest.fixture
def mock_process_controller():
    """Fixture providing mock process controller"""
    return MockProcessController()


@pytest.fixture
def process_manager(mock_process_controller):
    """Fixture providing ProcessManager with mock controller"""
    with patch('kwaainet_node.core.process_manager.PlatformFactory.create_process_controller') as mock_factory:
        mock_factory.return_value = mock_process_controller
        pm = ProcessManager(ResourceLimits(max_models=2))
        yield pm


@pytest.mark.asyncio
async def test_start_petals_process(process_manager):
    """Test starting a Petals process"""
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    blocks = [5, 6, 7, 8]

    # Start process
    process = await process_manager.start_petals_process(model_name, blocks)

    assert process.model_name == model_name
    assert process.blocks == blocks
    assert process.state == ProcessState.STARTING
    assert process.handle.pid == 10000

    # Wait for process to transition to running
    await asyncio.sleep(0.1)  # Allow monitoring task to run

    # Check it's in the manager's process list
    assert model_name in process_manager.processes
    assert process_manager.processes[model_name] == process


@pytest.mark.asyncio
async def test_start_duplicate_process(process_manager):
    """Test starting a process for a model that's already running"""
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    blocks = [5, 6, 7, 8]

    # Start first process
    process1 = await process_manager.start_petals_process(model_name, blocks)
    process1.state = ProcessState.RUNNING

    # Try to start same model again
    process2 = await process_manager.start_petals_process(model_name, blocks)

    # Should return the existing process
    assert process1 == process2


@pytest.mark.asyncio
async def test_resource_limits(process_manager):
    """Test resource limit enforcement"""
    # Process manager has max_models=2

    # Start two processes (should succeed)
    process1 = await process_manager.start_petals_process("model1", [1, 2])
    process2 = await process_manager.start_petals_process("model2", [3, 4])

    assert len(process_manager.processes) == 2

    # Try to start third process (should fail)
    with pytest.raises(RuntimeError, match="Maximum model limit"):
        await process_manager.start_petals_process("model3", [5, 6])


@pytest.mark.asyncio
async def test_stop_process(process_manager):
    """Test stopping a process"""
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    blocks = [5, 6, 7, 8]

    # Start process
    process = await process_manager.start_petals_process(model_name, blocks)
    assert model_name in process_manager.processes

    # Stop process
    success = await process_manager.stop_process(model_name)

    assert success
    assert process.state == ProcessState.STOPPED
    assert model_name not in process_manager.processes


@pytest.mark.asyncio
async def test_stop_nonexistent_process(process_manager):
    """Test stopping a process that doesn't exist"""
    success = await process_manager.stop_process("nonexistent-model")
    assert success is False


@pytest.mark.asyncio
async def test_list_processes(process_manager):
    """Test listing all processes"""
    # Initially empty
    processes = await process_manager.list_processes()
    assert len(processes) == 0

    # Add some processes
    await process_manager.start_petals_process("model1", [1, 2])
    await process_manager.start_petals_process("model2", [3, 4])

    processes = await process_manager.list_processes()
    assert len(processes) == 2
    assert "model1" in processes
    assert "model2" in processes


@pytest.mark.asyncio
async def test_ensure_model_available(process_manager):
    """Test ensure_model_available method"""
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    blocks = [5, 6, 7, 8]

    # First call should start the process
    process1 = await process_manager.ensure_model_available(model_name, blocks)
    assert process1.model_name == model_name

    # Set process to running and healthy
    process1.state = ProcessState.RUNNING
    process1.health_info = Mock()
    process1.health_info.status = Mock()
    process1.health_info.status.value = "healthy"

    # Second call should return existing process
    process2 = await process_manager.ensure_model_available(model_name, blocks)
    assert process1 == process2


@pytest.mark.asyncio
async def test_shutdown(process_manager):
    """Test shutting down all processes"""
    # Start some processes
    await process_manager.start_petals_process("model1", [1, 2])
    await process_manager.start_petals_process("model2", [3, 4])

    assert len(process_manager.processes) == 2

    # Shutdown
    await process_manager.shutdown()

    # All processes should be stopped and cleaned up
    assert len(process_manager.processes) == 0


@pytest.mark.asyncio
async def test_build_petals_command(process_manager):
    """Test building Petals command"""
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    blocks = [5, 6, 7, 8]
    port = 8080

    cmd = process_manager._build_petals_command(model_name, blocks, port)

    assert "python" in cmd
    assert "-m" in cmd
    assert "petals.cli.run_server" in cmd
    assert model_name in cmd
    assert "--block_indices" in cmd
    assert "5:9" in cmd  # min:max+1 format
    assert "--port" in cmd
    assert "8080" in cmd


if __name__ == "__main__":
    asyncio.run(pytest.main([__file__, "-v"]))