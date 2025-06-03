"""
自动对话多智能体系统 - 完整实现
集成Verilog工具链、持久化存储、监控可视化、高级学习机制
"""

import asyncio
import json
import time
import os
import sqlite3
import logging
import yaml
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import pickle
import subprocess
import tempfile
import shutil
from datetime import datetime
import threading

# 导入基础框架
from unified_autonomous_framework import *
from llm_integration import *
from utils.metrics import get_metrics_collector, MetricsCollector
from utils.logger import setup_logger

# ============================================================================
# 1. 完整的Verilog工具链集成
# ============================================================================

class VerilogTool(Enum):
    """Verilog工具类型"""
    ICARUS = "icarus"
    VERILATOR = "verilator"
    YOSYS = "yosys"
    GTKWAVE = "gtkwave"
    NEXTPNR = "nextpnr"

@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    files_generated: List[str] = field(default_factory=list)

class VerilogToolchain:
    """完整的Verilog工具链"""
    
    def __init__(self, work_dir: str = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="verilog_"))
        self.work_dir.mkdir(exist_ok=True, parents=True)
        self.logger = setup_logger(f"VerilogToolchain")
        
        # 检测可用工具
        self.available_tools = self._detect_tools()
        self.logger.info(f"Available tools: {list(self.available_tools.keys())}")
    
    def _detect_tools(self) -> Dict[VerilogTool, str]:
        """检测系统中可用的Verilog工具"""
        tools = {}
        
        tool_commands = {
            VerilogTool.ICARUS: ["iverilog", "--version"],
            VerilogTool.VERILATOR: ["verilator", "--version"],
            VerilogTool.YOSYS: ["yosys", "-V"],
            VerilogTool.GTKWAVE: ["gtkwave", "--version"],
            VerilogTool.NEXTPNR: ["nextpnr-ice40", "--version"]
        }
        
        for tool, cmd in tool_commands.items():
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=5)
                if result.returncode == 0:
                    tools[tool] = cmd[0]
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        
        return tools
    
    async def compile_verilog(self, code: str, module_name: str, 
                            include_dirs: List[str] = None) -> ToolResult:
        """编译Verilog代码"""
        start_time = time.time()
        
        # 写入源文件
        source_file = self.work_dir / f"{module_name}.v"
        with open(source_file, 'w') as f:
            f.write(code)
        
        if VerilogTool.ICARUS in self.available_tools:
            return await self._compile_with_icarus(source_file, module_name, include_dirs)
        elif VerilogTool.VERILATOR in self.available_tools:
            return await self._compile_with_verilator(source_file, module_name, include_dirs)
        else:
            return ToolResult(
                success=False,
                stdout="",
                stderr="No Verilog compiler available",
                return_code=-1,
                execution_time=time.time() - start_time
            )
    
    async def _compile_with_icarus(self, source_file: Path, module_name: str, 
                                 include_dirs: List[str] = None) -> ToolResult:
        """使用Icarus Verilog编译"""
        start_time = time.time()
        output_file = self.work_dir / f"{module_name}.vvp"
        
        cmd = ["iverilog", "-o", str(output_file), str(source_file)]
        
        if include_dirs:
            for inc_dir in include_dirs:
                cmd.extend(["-I", inc_dir])
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir
            )
            
            stdout, stderr = await process.communicate()
            
            return ToolResult(
                success=process.returncode == 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                return_code=process.returncode,
                execution_time=time.time() - start_time,
                files_generated=[str(output_file)] if process.returncode == 0 else []
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Compilation error: {str(e)}",
                return_code=-1,
                execution_time=time.time() - start_time
            )
    
    async def _compile_with_verilator(self, source_file: Path, module_name: str, 
                                    include_dirs: List[str] = None) -> ToolResult:
        """使用Verilator编译"""
        start_time = time.time()
        
        cmd = ["verilator", "--lint-only", str(source_file)]
        
        if include_dirs:
            for inc_dir in include_dirs:
                cmd.extend(["-I", inc_dir])
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir
            )
            
            stdout, stderr = await process.communicate()
            
            return ToolResult(
                success=process.returncode == 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                return_code=process.returncode,
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Verilator error: {str(e)}",
                return_code=-1,
                execution_time=time.time() - start_time
            )
    
    async def simulate_verilog(self, testbench_code: str, module_name: str, 
                             simulation_time: int = 1000) -> ToolResult:
        """仿真Verilog代码"""
        start_time = time.time()
        
        # 写入测试台文件
        testbench_file = self.work_dir / f"{module_name}_tb.v"
        with open(testbench_file, 'w') as f:
            f.write(testbench_code)
        
        if VerilogTool.ICARUS in self.available_tools:
            return await self._simulate_with_icarus(testbench_file, module_name, simulation_time)
        else:
            return ToolResult(
                success=False,
                stdout="",
                stderr="No Verilog simulator available",
                return_code=-1,
                execution_time=time.time() - start_time
            )
    
    async def _simulate_with_icarus(self, testbench_file: Path, module_name: str, 
                                  simulation_time: int) -> ToolResult:
        """使用Icarus Verilog仿真"""
        start_time = time.time()
        vvp_file = self.work_dir / f"{module_name}_tb.vvp"
        vcd_file = self.work_dir / f"{module_name}.vcd"
        
        # 编译测试台
        compile_cmd = ["iverilog", "-o", str(vvp_file), str(testbench_file)]
        
        try:
            # 编译
            compile_process = await asyncio.create_subprocess_exec(
                *compile_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir
            )
            
            compile_stdout, compile_stderr = await compile_process.communicate()
            
            if compile_process.returncode != 0:
                return ToolResult(
                    success=False,
                    stdout=compile_stdout.decode(),
                    stderr=compile_stderr.decode(),
                    return_code=compile_process.returncode,
                    execution_time=time.time() - start_time
                )
            
            # 运行仿真
            sim_cmd = ["vvp", str(vvp_file)]
            
            sim_process = await asyncio.create_subprocess_exec(
                *sim_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    sim_process.communicate(),
                    timeout=30  # 30秒超时
                )
            except asyncio.TimeoutError:
                sim_process.kill()
                return ToolResult(
                    success=False,
                    stdout="",
                    stderr="Simulation timeout",
                    return_code=-1,
                    execution_time=time.time() - start_time
                )
            
            files_generated = []
            if vcd_file.exists():
                files_generated.append(str(vcd_file))
            
            return ToolResult(
                success=sim_process.returncode == 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                return_code=sim_process.returncode,
                execution_time=time.time() - start_time,
                files_generated=files_generated
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Simulation error: {str(e)}",
                return_code=-1,
                execution_time=time.time() - start_time
            )
    
    async def synthesize_verilog(self, code: str, module_name: str, 
                               target: str = "ice40") -> ToolResult:
        """综合Verilog代码"""
        start_time = time.time()
        
        if VerilogTool.YOSYS not in self.available_tools:
            return ToolResult(
                success=False,
                stdout="",
                stderr="Yosys synthesizer not available",
                return_code=-1,
                execution_time=time.time() - start_time
            )
        
        # 写入源文件
        source_file = self.work_dir / f"{module_name}.v"
        with open(source_file, 'w') as f:
            f.write(code)
        
        # 创建Yosys脚本
        yosys_script = f"""
read_verilog {source_file}
hierarchy -top {module_name}
proc; opt; memory; opt
techmap; opt
synth_{target}
write_json {module_name}_synth.json
stat
"""
        
        script_file = self.work_dir / f"{module_name}_synth.ys"
        with open(script_file, 'w') as f:
            f.write(yosys_script)
        
        # 运行综合
        cmd = ["yosys", "-s", str(script_file)]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.work_dir
            )
            
            stdout, stderr = await process.communicate()
            
            files_generated = []
            json_file = self.work_dir / f"{module_name}_synth.json"
            if json_file.exists():
                files_generated.append(str(json_file))
            
            return ToolResult(
                success=process.returncode == 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                return_code=process.returncode,
                execution_time=time.time() - start_time,
                files_generated=files_generated
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Synthesis error: {str(e)}",
                return_code=-1,
                execution_time=time.time() - start_time
            )
    
    def cleanup(self):
        """清理工作目录"""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)

# ============================================================================
# 2. 持久化存储系统
# ============================================================================

class PersistenceManager:
    """持久化管理器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # 数据库连接
        self.db_path = self.data_dir / "system.db"
        self.logger = setup_logger("PersistenceManager")
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 智能体表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    profile TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 任务表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE NOT NULL,
                    task_data TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')
            
            # 对话表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT UNIQUE NOT NULL,
                    participants TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')
            
            # 学习记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    experience_type TEXT NOT NULL,
                    experience_data TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 知识库表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    knowledge_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_agent_profile(self, agent_name: str, profile: UnifiedAgentProfile):
        """保存智能体配置"""
        try:
            profile_data = json.dumps(asdict(profile), default=str)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO agents (name, profile, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (agent_name, profile_data))
                conn.commit()
            
            self.logger.info(f"Saved profile for agent: {agent_name}")
        
        except Exception as e:
            self.logger.error(f"Failed to save agent profile {agent_name}: {e}")
    
    def load_agent_profile(self, agent_name: str) -> Optional[UnifiedAgentProfile]:
        """加载智能体配置"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT profile FROM agents WHERE name = ?', (agent_name,))
                result = cursor.fetchone()
                
                if result:
                    profile_data = json.loads(result[0])
                    # 重构为UnifiedAgentProfile对象
                    return self._reconstruct_agent_profile(profile_data)
                
                return None
        
        except Exception as e:
            self.logger.error(f"Failed to load agent profile {agent_name}: {e}")
            return None
    
    def _reconstruct_agent_profile(self, data: Dict[str, Any]) -> UnifiedAgentProfile:
        """重构智能体配置对象"""
        # 转换枚举类型
        if 'role' in data:
            data['role'] = AgentRole(data['role'])
        
        if 'communication_style' in data:
            data['communication_style'] = CommunicationStyle(data['communication_style'])
        
        if 'personality_traits' in data:
            traits = {}
            for trait, value in data['personality_traits'].items():
                try:
                    trait_enum = PersonalityTrait(trait)
                    traits[trait_enum] = value
                except ValueError:
                    continue
            data['personality_traits'] = traits
        
        if 'capabilities' in data:
            caps = {}
            for cap, value in data['capabilities'].items():
                try:
                    cap_enum = AgentCapability(cap)
                    caps[cap_enum] = value
                except ValueError:
                    continue
            data['capabilities'] = caps
        
        return UnifiedAgentProfile(**data)
    
    def save_task(self, task: UnifiedTask):
        """保存任务"""
        try:
            task_data = json.dumps(asdict(task), default=str)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO tasks (task_id, task_data, status, completed_at)
                    VALUES (?, ?, ?, ?)
                ''', (task.task_id, task_data, task.status, 
                     task.completed_at if task.completed_at else None))
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to save task {task.task_id}: {e}")
    
    def save_conversation(self, conversation_id: str, participants: List[str], 
                         messages: List[Dict[str, Any]]):
        """保存对话"""
        try:
            participants_data = json.dumps(participants)
            messages_data = json.dumps(messages, default=str)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO conversations 
                    (conversation_id, participants, messages)
                    VALUES (?, ?, ?)
                ''', (conversation_id, participants_data, messages_data))
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation {conversation_id}: {e}")
    
    def save_learning_record(self, agent_name: str, experience_type: str, 
                           experience_data: Dict[str, Any], success: bool):
        """保存学习记录"""
        try:
            data_json = json.dumps(experience_data, default=str)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO learning_records 
                    (agent_name, experience_type, experience_data, success)
                    VALUES (?, ?, ?, ?)
                ''', (agent_name, experience_type, data_json, success))
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to save learning record: {e}")
    
    def save_knowledge(self, knowledge_type: str, content: str, 
                      metadata: Dict[str, Any] = None):
        """保存知识"""
        try:
            metadata_json = json.dumps(metadata or {}, default=str)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO knowledge_base (knowledge_type, content, metadata)
                    VALUES (?, ?, ?)
                ''', (knowledge_type, content, metadata_json))
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to save knowledge: {e}")
    
    def get_agent_learning_history(self, agent_name: str, 
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """获取智能体学习历史"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT experience_type, experience_data, success, timestamp
                    FROM learning_records
                    WHERE agent_name = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (agent_name, limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'experience_type': row[0],
                        'experience_data': json.loads(row[1]),
                        'success': bool(row[2]),
                        'timestamp': row[3]
                    })
                
                return results
        
        except Exception as e:
            self.logger.error(f"Failed to get learning history for {agent_name}: {e}")
            return []

# ============================================================================
# 3. Web监控和可视化系统
# ============================================================================

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json

class WebMonitor:
    """Web监控系统"""
    
    def __init__(self, framework: 'AutoDialogueFramework', port: int = 5000):
        self.framework = framework
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'autonomous_agents_monitor'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.port = port
        
        self.logger = setup_logger("WebMonitor")
        self._setup_routes()
        self._setup_websocket_events()
    
    def _setup_routes(self):
        """设置Web路由"""
        
        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')
        
        @self.app.route('/api/system/status')
        def system_status():
            return jsonify(self.framework.get_system_status())
        
        @self.app.route('/api/agents')
        def get_agents():
            agents_data = {}
            for name, agent in self.framework.agents.items():
                agents_data[name] = {
                    'name': agent.profile.name,
                    'role': agent.profile.role.value,
                    'workload': agent.profile.current_workload,
                    'communication_style': agent.profile.communication_style.value,
                    'active_conversations': len(agent.active_conversations),
                    'success_rate': agent.profile.success_rate,
                    'last_activity': getattr(agent, 'last_activity_time', time.time())
                }
            return jsonify(agents_data)
        
        @self.app.route('/api/tasks')
        def get_tasks():
            return jsonify({
                'active_tasks': [asdict(task) for task in self.framework.system_coordinator.task_queue],
                'completed_tasks': len(self.framework.system_coordinator.completed_tasks)
            })
        
        @self.app.route('/api/conversations')
        def get_conversations():
            conversations = []
            for agent in self.framework.agents.values():
                for conv_id, conv_data in agent.active_conversations.items():
                    conversations.append({
                        'id': conv_id,
                        'participants': conv_data.get('partners', []),
                        'status': conv_data.get('status', 'active'),
                        'initiated_at': conv_data.get('initiated_at', time.time())
                    })
            return jsonify(conversations)
        
        @self.app.route('/api/metrics')
        def get_metrics():
            metrics = get_metrics_collector()
            return jsonify({
                'system_metrics': metrics.get_system_summary(),
                'real_time_stats': metrics.get_real_time_stats()
            })
        
        @self.app.route('/api/submit_task', methods=['POST'])
        def submit_task():
            data = request.json
            try:
                task_id = asyncio.run(self.framework.submit_task(
                    description=data['description'],
                    task_type=data['task_type'],
                    requirements=data.get('requirements', {}),
                    priority=data.get('priority', 0.5)
                ))
                return jsonify({'success': True, 'task_id': task_id})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    
    def _setup_websocket_events(self):
        """设置WebSocket事件"""
        
        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info('Client connected to monitoring dashboard')
            emit('status', {'message': 'Connected to monitoring system'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info('Client disconnected from monitoring dashboard')
        
        @self.socketio.on('request_update')
        def handle_update_request():
            # 发送实时系统状态
            status = self.framework.get_system_status()
            emit('system_update', status)
    
    def start_monitoring(self):
        """启动监控服务器"""
        def run_server():
            self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False)
        
        monitor_thread = threading.Thread(target=run_server, daemon=True)
        monitor_thread.start()
        self.logger.info(f"Web monitor started on port {self.port}")
    
    def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """广播事件到所有连接的客户端"""
        self.socketio.emit('live_event', {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        })

# ============================================================================
# 4. 高级学习机制
# ============================================================================

class AdvancedLearningSystem:
    """高级学习系统"""
    
    def __init__(self, persistence_manager: PersistenceManager):
        self.persistence_manager = persistence_manager
        self.logger = setup_logger("AdvancedLearning")
        
        # 模式识别器
        self.pattern_recognizer = PatternRecognizer()
        
        # 知识迁移器
        self.knowledge_transfer = KnowledgeTransfer()
        
        # 集体智能管理器
        self.collective_intelligence = CollectiveIntelligence()
        
        # 强化学习组件
        self.reinforcement_learner = ReinforcementLearner()
    
    async def process_experience(self, agent_name: str, experience: Dict[str, Any]) -> Dict[str, Any]:
        """处理经验并提取学习insights"""
        
        # 1. 模式识别
        patterns = await self.pattern_recognizer.identify_patterns(agent_name, experience)
        
        # 2. 强化学习更新
        rl_insights = await self.reinforcement_learner.update_policy(agent_name, experience)
        
        # 3. 知识迁移机会
        transfer_opportunities = await self.knowledge_transfer.identify_transfer_opportunities(experience)
        
        # 4. 集体智能贡献
        collective_insights = await self.collective_intelligence.contribute_experience(agent_name, experience)
        
        # 保存学习记录
        self.persistence_manager.save_learning_record(
            agent_name, experience.get('type', 'general'), experience, 
            experience.get('success', False)
        )
        
        return {
            'patterns': patterns,
            'reinforcement_insights': rl_insights,
            'transfer_opportunities': transfer_opportunities,
            'collective_insights': collective_insights
        }

class PatternRecognizer:
    """模式识别器"""
    
    def __init__(self):
        self.logger = setup_logger("PatternRecognizer")
        self.patterns = defaultdict(list)
    
    async def identify_patterns(self, agent_name: str, experience: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别经验中的模式"""
        patterns_found = []
        
        # 成功模式识别
        if experience.get('success', False):
            pattern = self._extract_success_pattern(experience)
            if pattern:
                patterns_found.append({
                    'type': 'success_pattern',
                    'pattern': pattern,
                    'confidence': 0.8
                })
        
        # 失败模式识别
        else:
            pattern = self._extract_failure_pattern(experience)
            if pattern:
                patterns_found.append({
                    'type': 'failure_pattern',
                    'pattern': pattern,
                    'confidence': 0.7
                })
        
        # 时序模式识别
        temporal_pattern = self._extract_temporal_pattern(agent_name, experience)
        if temporal_pattern:
            patterns_found.append({
                'type': 'temporal_pattern',
                'pattern': temporal_pattern,
                'confidence': 0.6
            })
        
        return patterns_found
    
    def _extract_success_pattern(self, experience: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提取成功模式"""
        if experience.get('task_type') and experience.get('approach'):
            return {
                'task_type': experience['task_type'],
                'approach': experience['approach'],
                'duration': experience.get('duration', 0),
                'context': experience.get('context', {})
            }
        return None
    
    def _extract_failure_pattern(self, experience: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提取失败模式"""
        if experience.get('error_type'):
            return {
                'error_type': experience['error_type'],
                'task_type': experience.get('task_type'),
                'context': experience.get('context', {}),
                'error_message': experience.get('error_message', '')
            }
        return None
    
    def _extract_temporal_pattern(self, agent_name: str, experience: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提取时序模式"""
        # 简化实现：检查最近的经验序列
        recent_experiences = self.patterns[agent_name][-10:]  # 最近10个经验
        
        if len(recent_experiences) >= 3:
            # 检查是否有重复的成功/失败序列
            success_sequence = [exp.get('success', False) for exp in recent_experiences[-3:]]
            if all(success_sequence) or not any(success_sequence):
                return {
                    'sequence_type': 'success_streak' if all(success_sequence) else 'failure_streak',
                    'length': len(success_sequence),
                    'pattern': success_sequence
                }
        
        return None

class KnowledgeTransfer:
    """知识迁移系统"""
    
    def __init__(self):
        self.logger = setup_logger("KnowledgeTransfer")
        self.transfer_history = defaultdict(list)
    
    async def identify_transfer_opportunities(self, experience: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别知识迁移机会"""
        opportunities = []
        
        # 任务类型相似性迁移
        if experience.get('task_type') and experience.get('success', False):
            similar_tasks = self._find_similar_tasks(experience['task_type'])
            for similar_task in similar_tasks:
                opportunities.append({
                    'type': 'task_similarity_transfer',
                    'source_task': experience['task_type'],
                    'target_task': similar_task,
                    'transferable_knowledge': experience.get('approach', ''),
                    'confidence': 0.7
                })
        
        # 错误处理迁移
        if experience.get('error_type') and experience.get('solution'):
            opportunities.append({
                'type': 'error_handling_transfer',
                'error_type': experience['error_type'],
                'solution': experience['solution'],
                'confidence': 0.8
            })
        
        return opportunities
    
    def _find_similar_tasks(self, task_type: str) -> List[str]:
        """找到相似的任务类型"""
        # 简化的相似性判断
        similarity_map = {
            'simple_logic': ['combinational'],
            'combinational': ['simple_logic', 'sequential'],
            'sequential': ['combinational', 'complex_module'],
            'complex_module': ['sequential'],
            'testbench': ['simulation'],
            'optimization': ['complex_module']
        }
        
        return similarity_map.get(task_type, [])

class CollectiveIntelligence:
    """集体智能系统"""
    
    def __init__(self):
        self.logger = setup_logger("CollectiveIntelligence")
        self.collective_knowledge = defaultdict(dict)
        self.consensus_tracking = defaultdict(list)
    
    async def contribute_experience(self, agent_name: str, experience: Dict[str, Any]) -> Dict[str, Any]:
        """向集体智能贡献经验"""
        contribution = {
            'contributor': agent_name,
            'experience': experience,
            'timestamp': time.time()
        }
        
        # 按任务类型分类
        task_type = experience.get('task_type', 'general')
        if task_type not in self.collective_knowledge:
            self.collective_knowledge[task_type] = {
                'successful_approaches': defaultdict(int),
                'common_errors': defaultdict(int),
                'best_practices': [],
                'collective_insights': []
            }
        
        # 更新集体知识
        if experience.get('success', False):
            approach = experience.get('approach', 'unknown')
            self.collective_knowledge[task_type]['successful_approaches'][approach] += 1
        
        if experience.get('error_type'):
            error_type = experience['error_type']
            self.collective_knowledge[task_type]['common_errors'][error_type] += 1
        
        # 生成集体洞察
        insights = self._generate_collective_insights(task_type)
        
        return {
            'contribution_accepted': True,
            'collective_insights': insights,
            'knowledge_impact': self._calculate_knowledge_impact(contribution)
        }
    
    def _generate_collective_insights(self, task_type: str) -> List[Dict[str, Any]]:
        """生成集体洞察"""
        insights = []
        knowledge = self.collective_knowledge.get(task_type, {})
        
        # 最佳方法洞察
        if knowledge.get('successful_approaches'):
            best_approach = max(
                knowledge['successful_approaches'].items(),
                key=lambda x: x[1]
            )
            insights.append({
                'type': 'best_approach',
                'task_type': task_type,
                'approach': best_approach[0],
                'success_count': best_approach[1],
                'confidence': min(0.9, best_approach[1] / 10.0)
            })
        
        # 常见错误洞察
        if knowledge.get('common_errors'):
            most_common_error = max(
                knowledge['common_errors'].items(),
                key=lambda x: x[1]
            )
            insights.append({
                'type': 'common_error_warning',
                'task_type': task_type,
                'error_type': most_common_error[0],
                'occurrence_count': most_common_error[1],
                'priority': 'high' if most_common_error[1] > 5 else 'medium'
            })
        
        return insights
    
    def _calculate_knowledge_impact(self, contribution: Dict[str, Any]) -> float:
        """计算知识贡献的影响力"""
        # 简化的影响力计算
        base_impact = 0.1
        
        experience = contribution['experience']
        
        # 成功经验有更高影响力
        if experience.get('success', False):
            base_impact += 0.3
        
        # 新的错误类型有较高影响力
        if experience.get('error_type') and experience.get('solution'):
            base_impact += 0.2
        
        # 复杂任务的经验有更高影响力
        if experience.get('task_complexity', 0) > 0.7:
            base_impact += 0.2
        
        return min(1.0, base_impact)

class ReinforcementLearner:
    """强化学习组件"""
    
    def __init__(self):
        self.logger = setup_logger("ReinforcementLearner")
        self.q_tables = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
    
    async def update_policy(self, agent_name: str, experience: Dict[str, Any]) -> Dict[str, Any]:
        """更新强化学习策略"""
        state = self._extract_state(experience)
        action = self._extract_action(experience)
        reward = self._calculate_reward(experience)
        next_state = self._extract_next_state(experience)
        
        # Q-learning更新
        current_q = self.q_tables[agent_name][(state, action)]
        max_next_q = max(self.q_tables[agent_name][(next_state, a)] for a in self._get_possible_actions(next_state)) if next_state else 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_tables[agent_name][(state, action)] = new_q
        
        return {
            'q_value_updated': True,
            'old_q_value': current_q,
            'new_q_value': new_q,
            'reward': reward,
            'policy_insight': self._get_policy_insight(agent_name, state)
        }
    
    def _extract_state(self, experience: Dict[str, Any]) -> str:
        """从经验中提取状态"""
        return f"{experience.get('task_type', 'unknown')}_{experience.get('complexity', 0.5):.1f}"
    
    def _extract_action(self, experience: Dict[str, Any]) -> str:
        """从经验中提取动作"""
        return experience.get('approach', 'default')
    
    def _calculate_reward(self, experience: Dict[str, Any]) -> float:
        """计算奖励"""
        base_reward = 1.0 if experience.get('success', False) else -0.5
        
        # 时间奖励/惩罚
        duration = experience.get('duration', 0)
        if duration > 0:
            time_factor = max(0.1, 1.0 - duration / 300.0)  # 5分钟为基准
            base_reward *= time_factor
        
        # 质量奖励
        if experience.get('quality_score'):
            quality_bonus = (experience['quality_score'] - 0.5) * 0.5
            base_reward += quality_bonus
        
        return base_reward
    
    def _extract_next_state(self, experience: Dict[str, Any]) -> Optional[str]:
        """提取下一个状态"""
        # 简化实现：如果任务完成，下一状态为None
        if experience.get('task_completed', False):
            return None
        return self._extract_state(experience)
    
    def _get_possible_actions(self, state: str) -> List[str]:
        """获取状态下的可能动作"""
        return ['standard', 'creative', 'analytical', 'systematic']
    
    def _get_policy_insight(self, agent_name: str, state: str) -> Dict[str, Any]:
        """获取策略洞察"""
        q_values = {action: self.q_tables[agent_name][(state, action)] 
                   for action in self._get_possible_actions(state)}
        
        best_action = max(q_values.keys(), key=lambda k: q_values[k])
        
        return {
            'recommended_action': best_action,
            'confidence': max(q_values.values()) if q_values.values() else 0.0,
            'action_values': q_values
        }

# ============================================================================
# 5. 自动对话主框架
# ============================================================================

class AutoDialogueFramework(UnifiedAutonomousFramework):
    """自动对话多智能体框架"""
    
    def __init__(self, config_path: str = None, max_dialogue_rounds: int = 10):
        super().__init__()
        
        self.max_dialogue_rounds = max_dialogue_rounds
        self.config_path = config_path
        
        # 集成组件
        self.toolchain = VerilogToolchain()
        self.persistence_manager = PersistenceManager()
        self.learning_system = AdvancedLearningSystem(self.persistence_manager)
        self.web_monitor = WebMonitor(self, port=5000)
        
        # 对话管理
        self.active_dialogues: Dict[str, 'AutoDialogue'] = {}
        self.dialogue_history = deque(maxlen=100)
        
        # 配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        # LLM集成
        if 'llm' in self.config:
            self.llm_manager = LLMManager(LLMConfig(**self.config['llm']))
        
        self.logger = setup_logger("AutoDialogueFramework")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 2000
            },
            'agents': {
                'coder': {
                    'personality': {'creative': 0.8, 'proactive': 0.7},
                    'capabilities': ['code_generation', 'debugging']
                },
                'reviewer': {
                    'personality': {'analytical': 0.9, 'detail_oriented': 0.9},
                    'capabilities': ['code_review', 'error_analysis']
                },
                'executor': {
                    'personality': {'focused': 0.9, 'reliable': 0.9},
                    'capabilities': ['compilation', 'simulation']
                }
            },
            'dialogue': {
                'max_rounds': 10,
                'timeout_minutes': 30,
                'auto_escalation': True
            }
        }
    
    async def initialize(self):
        """初始化框架"""
        await super().initialize()
        
        # 启动Web监控
        self.web_monitor.start_monitoring()
        
        # 加载持久化的智能体配置
        await self._load_persistent_agents()
        
        self.logger.info("AutoDialogueFramework initialized with all components")
    
    async def _load_persistent_agents(self):
        """加载持久化的智能体配置"""
        for agent_name in self.agents.keys():
            saved_profile = self.persistence_manager.load_agent_profile(agent_name)
            if saved_profile:
                # 更新智能体配置
                self.agents[agent_name].profile = saved_profile
                self.logger.info(f"Loaded saved profile for agent: {agent_name}")
    
    async def start_auto_dialogue(self, problem_description: str, test_code: str = None, 
                                custom_prompt: str = None) -> str:
        """启动自动对话"""
        dialogue_id = f"dialogue_{int(time.time())}_{len(self.active_dialogues)}"
        
        dialogue = AutoDialogue(
            dialogue_id=dialogue_id,
            framework=self,
            problem_description=problem_description,
            test_code=test_code,
            custom_prompt=custom_prompt,
            max_rounds=self.max_dialogue_rounds
        )
        
        self.active_dialogues[dialogue_id] = dialogue
        
        # 启动对话
        result = await dialogue.run()
        
        # 移动到历史记录
        self.dialogue_history.append(dialogue)
        del self.active_dialogues[dialogue_id]
        
        # 保存对话记录
        self.persistence_manager.save_conversation(
            dialogue_id, dialogue.participants, dialogue.get_message_history()
        )
        
        # 广播事件
        self.web_monitor.broadcast_event('dialogue_completed', {
            'dialogue_id': dialogue_id,
            'success': result['success'],
            'rounds': result['rounds'],
            'final_code': result.get('final_code', '')
        })
        
        return dialogue_id

class AutoDialogue:
    """自动对话管理器"""
    
    def __init__(self, dialogue_id: str, framework: AutoDialogueFramework,
                 problem_description: str, test_code: str = None,
                 custom_prompt: str = None, max_rounds: int = 10):
        
        self.dialogue_id = dialogue_id
        self.framework = framework
        self.problem_description = problem_description
        self.test_code = test_code
        self.custom_prompt = custom_prompt
        self.max_rounds = max_rounds
        
        self.current_round = 0
        self.participants = []
        self.message_history: List[Dict[str, Any]] = []
        self.current_code = ""
        self.task_completed = False
        self.success = False
        
        self.logger = setup_logger(f"AutoDialogue.{dialogue_id}")
        
        # 获取参与智能体
        self.coder_agent = self._select_agent_by_role(AgentRole.CODE_GENERATOR)
        self.reviewer_agent = self._select_agent_by_role(AgentRole.CODE_REVIEWER)
        self.executor_agent = self._select_agent_by_role(AgentRole.CODE_EXECUTOR)
        
        if self.coder_agent:
            self.participants.append(self.coder_agent.profile.name)
        if self.reviewer_agent:
            self.participants.append(self.reviewer_agent.profile.name)
        if self.executor_agent:
            self.participants.append(self.executor_agent.profile.name)
    
    def _select_agent_by_role(self, role: AgentRole) -> Optional[AutonomousAgent]:
        """根据角色选择智能体"""
        for agent in self.framework.agents.values():
            if agent.profile.role == role:
                return agent
        return None
    
    async def run(self) -> Dict[str, Any]:
        """运行自动对话"""
        self.logger.info(f"Starting auto dialogue: {self.dialogue_id}")
        
        # 记录对话开始
        metrics = get_metrics_collector()
        metrics.start_experiment(self.dialogue_id)
        
        try:
            # 初始化对话
            await self._initialize_dialogue()
            
            # 主对话循环
            while self.current_round < self.max_rounds and not self.task_completed:
                self.current_round += 1
                self.logger.info(f"Round {self.current_round}/{self.max_rounds}")
                
                # 执行对话轮次
                round_result = await self._execute_round()
                
                # 检查完成条件
                if round_result.get('task_completed', False):
                    self.task_completed = True
                    self.success = round_result.get('success', False)
                    self.current_code = round_result.get('final_code', self.current_code)
                    break
                
                # 检查是否需要干预
                if await self._needs_intervention():
                    await self._handle_intervention()
            
            # 生成最终结果
            result = await self._generate_final_result()
            
            # 记录学习经验
            await self._record_learning_experiences(result)
            
            # 记录对话结束
            metrics.finish_experiment(
                self.dialogue_id, 
                self.success, 
                self.current_code if self.success else None,
                None if self.success else "Task not completed"
            )
            
            return result
        
        except Exception as e:
            self.logger.error(f"Dialogue failed: {e}")
            metrics.finish_experiment(self.dialogue_id, False, error_message=str(e))
            return {
                'success': False,
                'error': str(e),
                'rounds': self.current_round,
                'dialogue_id': self.dialogue_id
            }
    
    async def _initialize_dialogue(self):
        """初始化对话"""
        # 创建初始任务消息
        initial_prompt = self._build_initial_prompt()
        
        initial_message = {
            'sender': 'system',
            'content': initial_prompt,
            'timestamp': time.time(),
            'round': 0,
            'type': 'task_initialization'
        }
        
        self.message_history.append(initial_message)
        
        # 通知所有参与者
        for participant in self.participants:
            if participant in self.framework.agents:
                agent = self.framework.agents[participant]
                await self._send_message_to_agent(agent, initial_message)
    
    def _build_initial_prompt(self) -> str:
        """构建初始提示"""
        prompt_parts = [
            f"Task: {self.problem_description}",
            "",
            "You are participating in an autonomous multi-agent design process.",
            "Work collaboratively to create a working Verilog design."
        ]
        
        if self.test_code:
            prompt_parts.extend([
                "",
                "Test code provided:",
                "```verilog",
                self.test_code,
                "```"
            ])
        
        if self.custom_prompt:
            prompt_parts.extend([
                "",
                "Additional requirements:",
                self.custom_prompt
            ])
        
        prompt_parts.extend([
            "",
            "Success criteria:",
            "1. Code compiles without errors",
            "2. Simulation runs successfully",
            "3. All test cases pass",
            "",
            "Begin the design process."
        ])
        
        return "\n".join(prompt_parts)
    
    async def _execute_round(self) -> Dict[str, Any]:
        """执行一轮对话"""
        round_result = {
            'round': self.current_round,
            'task_completed': False,
            'success': False,
            'actions_taken': []
        }
        
        # 1. Coder生成/改进代码
        if self.coder_agent:
            coder_result = await self._agent_code_generation()
            round_result['actions_taken'].append(coder_result)
            
            if coder_result.get('code_generated'):
                self.current_code = coder_result['code']
        
        # 2. Reviewer审查代码
        if self.reviewer_agent and self.current_code:
            review_result = await self._agent_code_review()
            round_result['actions_taken'].append(review_result)
            
            if not review_result.get('approved', False):
                # 需要修改，返回给Coder
                return round_result
        
        # 3. Executor执行测试
        if self.executor_agent and self.current_code:
            execution_result = await self._agent_code_execution()
            round_result['actions_taken'].append(execution_result)
            
            if execution_result.get('success', False):
                round_result['task_completed'] = True
                round_result['success'] = True
                round_result['final_code'] = self.current_code
        
        return round_result
    
    async def _agent_code_generation(self) -> Dict[str, Any]:
        """智能体代码生成"""
        if not self.coder_agent:
            return {'error': 'No coder agent available'}
        
        # 构建生成提示
        generation_context = {
            'problem': self.problem_description,
            'current_round': self.current_round,
            'previous_feedback': self._get_previous_feedback(),
            'test_code': self.test_code
        }
        
        # 使用LLM生成代码
        if hasattr(self.framework, 'llm_manager'):
            try:
                code = await self.framework.llm_manager.generate_code(
                    task_type="verilog_design",
                    requirements=generation_context,
                    agent_profile={
                        'name': self.coder_agent.profile.name,
                        'personality_traits': {
                            k.value: v for k, v in self.coder_agent.profile.personality_traits.items()
                        }
                    }
                )
                
                # 记录消息
                message = {
                    'sender': self.coder_agent.profile.name,
                    'content': f"Generated code:\n```verilog\n{code}\n```",
                    'timestamp': time.time(),
                    'round': self.current_round,
                    'type': 'code_generation'
                }
                
                self.message_history.append(message)
                
                # 记录指标
                metrics = get_metrics_collector()
                metrics.record_code_generation(self.dialogue_id, self.coder_agent.profile.name, True, self.current_round)
                
                return {
                    'action': 'code_generation',
                    'agent': self.coder_agent.profile.name,
                    'code_generated': True,
                    'code': code,
                    'success': True
                }
            
            except Exception as e:
                self.logger.error(f"Code generation failed: {e}")
                
                # 记录错误
                metrics = get_metrics_collector()
                metrics.record_error(self.dialogue_id, 'code_generation_error', str(e), self.coder_agent.profile.name)
                
                return {
                    'action': 'code_generation',
                    'agent': self.coder_agent.profile.name,
                    'code_generated': False,
                    'error': str(e),
                    'success': False
                }
        
        return {'error': 'LLM manager not available'}
    
    async def _agent_code_review(self) -> Dict[str, Any]:
        """智能体代码审查"""
        if not self.reviewer_agent:
            return {'error': 'No reviewer agent available'}
        
        # 使用LLM审查代码
        if hasattr(self.framework, 'llm_manager'):
            try:
                review_result = await self.framework.llm_manager.review_code(
                    code=self.current_code,
                    requirements={'problem': self.problem_description},
                    reviewer_profile={
                        'name': self.reviewer_agent.profile.name,
                        'personality_traits': {
                            k.value: v for k, v in self.reviewer_agent.profile.personality_traits.items()
                        }
                    }
                )
                
                approved = review_result.get('status') == 'approved'
                issues = review_result.get('issues', [])
                suggestions = review_result.get('suggestions', [])
                
                # 记录消息
                review_summary = f"Review result: {'APPROVED' if approved else 'NEEDS IMPROVEMENT'}"
                if issues:
                    review_summary += f"\nIssues found: {len(issues)}"
                    for issue in issues[:3]:  # 只显示前3个问题
                        review_summary += f"\n- {issue.get('description', 'Unknown issue')}"
                
                message = {
                    'sender': self.reviewer_agent.profile.name,
                    'content': review_summary,
                    'timestamp': time.time(),
                    'round': self.current_round,
                    'type': 'code_review',
                    'review_data': review_result
                }
                
                self.message_history.append(message)
                
                return {
                    'action': 'code_review',
                    'agent': self.reviewer_agent.profile.name,
                    'approved': approved,
                    'issues': issues,
                    'suggestions': suggestions,
                    'success': True
                }
            
            except Exception as e:
                self.logger.error(f"Code review failed: {e}")
                
                # 记录错误
                metrics = get_metrics_collector()
                metrics.record_error(self.dialogue_id, 'code_review_error', str(e), self.reviewer_agent.profile.name)
                
                return {
                    'action': 'code_review',
                    'agent': self.reviewer_agent.profile.name,
                    'approved': False,
                    'error': str(e),
                    'success': False
                }
        
        return {'error': 'LLM manager not available'}
    
    async def _agent_code_execution(self) -> Dict[str, Any]:
        """智能体代码执行"""
        if not self.executor_agent:
            return {'error': 'No executor agent available'}
        
        try:
            # 提取模块名
            module_name = self._extract_module_name(self.current_code)
            
            # 编译代码
            compile_result = await self.framework.toolchain.compile_verilog(
                self.current_code, module_name
            )
            
            metrics = get_metrics_collector()
            metrics.record_compilation(self.dialogue_id, compile_result.success, compile_result.stderr)
            
            if not compile_result.success:
                message = {
                    'sender': self.executor_agent.profile.name,
                    'content': f"Compilation failed:\n{compile_result.stderr}",
                    'timestamp': time.time(),
                    'round': self.current_round,
                    'type': 'compilation_result'
                }
                
                self.message_history.append(message)
                
                return {
                    'action': 'code_execution',
                    'agent': self.executor_agent.profile.name,
                    'compilation_success': False,
                    'compilation_error': compile_result.stderr,
                    'success': False
                }
            
            # 如果有测试代码，运行仿真
            if self.test_code:
                sim_result = await self.framework.toolchain.simulate_verilog(
                    self.test_code, module_name
                )
                
                metrics.record_simulation(self.dialogue_id, sim_result.success, sim_result.execution_time)
                
                sim_success = sim_result.success and "error" not in sim_result.stdout.lower()
                
                message = {
                    'sender': self.executor_agent.profile.name,
                    'content': f"Simulation {'PASSED' if sim_success else 'FAILED'}:\n{sim_result.stdout}",
                    'timestamp': time.time(),
                    'round': self.current_round,
                    'type': 'simulation_result'
                }
                
                self.message_history.append(message)
                
                return {
                    'action': 'code_execution',
                    'agent': self.executor_agent.profile.name,
                    'compilation_success': True,
                    'simulation_success': sim_success,
                    'simulation_output': sim_result.stdout,
                    'success': sim_success
                }
            else:
                # 没有测试代码，编译成功就算成功
                message = {
                    'sender': self.executor_agent.profile.name,
                    'content': "Compilation successful. No test code provided.",
                    'timestamp': time.time(),
                    'round': self.current_round,
                    'type': 'compilation_result'
                }
                
                self.message_history.append(message)
                
                return {
                    'action': 'code_execution',
                    'agent': self.executor_agent.profile.name,
                    'compilation_success': True,
                    'simulation_success': True,  # 假设成功
                    'success': True
                }
        
        except Exception as e:
            self.logger.error(f"Code execution failed: {e}")
            
            # 记录错误
            metrics = get_metrics_collector()
            metrics.record_error(self.dialogue_id, 'execution_error', str(e), self.executor_agent.profile.name)
            
            return {
                'action': 'code_execution',
                'agent': self.executor_agent.profile.name,
                'compilation_success': False,
                'error': str(e),
                'success': False
            }
    
    def _extract_module_name(self, code: str) -> str:
        """从代码中提取模块名"""
        import re
        match = re.search(r'\bmodule\s+(\w+)', code)
        return match.group(1) if match else "design"
    
    def _get_previous_feedback(self) -> List[str]:
        """获取之前的反馈"""
        feedback = []
        for message in self.message_history[-5:]:  # 最近5条消息
            if message.get('type') in ['code_review', 'compilation_result', 'simulation_result']:
                feedback.append(message['content'])
        return feedback
    
    async def _needs_intervention(self) -> bool:
        """检查是否需要干预"""
        # 如果连续3轮都失败，需要干预
        if self.current_round >= 3:
            recent_actions = []
            for message in self.message_history[-6:]:
                if message.get('type') in ['compilation_result', 'simulation_result']:
                    recent_actions.append('failed' in message['content'].lower() or 'error' in message['content'].lower())
            
            if len(recent_actions) >= 3 and all(recent_actions[-3:]):
                return True
        
        return False
    
    async def _handle_intervention(self):
        """处理干预"""
        self.logger.warning(f"Intervention needed for dialogue {self.dialogue_id}")
        
        # 生成干预消息
        intervention_message = {
            'sender': 'system',
            'content': (
                "INTERVENTION: Multiple failures detected. "
                "Please review the problem requirements and try a different approach. "
                "Consider simplifying the design or focusing on specific issues."
            ),
            'timestamp': time.time(),
            'round': self.current_round,
            'type': 'intervention'
        }
        
        self.message_history.append(intervention_message)
        
        # 重置某些状态以鼓励新方法
        if self.coder_agent:
            # 降低创造性，提高谨慎性
            self.coder_agent.profile.personality_traits[PersonalityTrait.CAUTIOUS] = 0.8
            self.coder_agent.profile.personality_traits[PersonalityTrait.CREATIVE] = 0.5
    
    async def _generate_final_result(self) -> Dict[str, Any]:
        """生成最终结果"""
        result = {
            'dialogue_id': self.dialogue_id,
            'success': self.success,
            'task_completed': self.task_completed,
            'rounds': self.current_round,
            'max_rounds': self.max_rounds,
            'participants': self.participants,
            'message_count': len(self.message_history),
            'final_code': self.current_code if self.success else None,
            'problem_description': self.problem_description
        }
        
        # 添加统计信息
        result['statistics'] = self._calculate_dialogue_statistics()
        
        # 添加学习洞察
        result['learning_insights'] = await self._extract_learning_insights()
        
        return result
    
    def _calculate_dialogue_statistics(self) -> Dict[str, Any]:
        """计算对话统计信息"""
        stats = {
            'total_messages': len(self.message_history),
            'messages_per_agent': defaultdict(int),
            'action_counts': defaultdict(int),
            'round_durations': [],
            'error_count': 0
        }
        
        for message in self.message_history:
            sender = message.get('sender', 'unknown')
            stats['messages_per_agent'][sender] += 1
            
            msg_type = message.get('type', 'unknown')
            stats['action_counts'][msg_type] += 1
            
            if 'error' in message.get('content', '').lower() or 'failed' in message.get('content', '').lower():
                stats['error_count'] += 1
        
        return dict(stats)
    
    async def _extract_learning_insights(self) -> List[Dict[str, Any]]:
        """提取学习洞察"""
        insights = []
        
        # 成功模式分析
        if self.success:
            insights.append({
                'type': 'success_pattern',
                'rounds_to_success': self.current_round,
                'key_factors': self._identify_success_factors()
            })
        
        # 失败模式分析
        else:
            insights.append({
                'type': 'failure_pattern',
                'failure_reasons': self._identify_failure_reasons(),
                'suggested_improvements': self._suggest_improvements()
            })
        
        # 协作效率分析
        insights.append({
            'type': 'collaboration_efficiency',
            'efficiency_score': self._calculate_efficiency_score(),
            'bottlenecks': self._identify_bottlenecks()
        })
        
        return insights
    
    def _identify_success_factors(self) -> List[str]:
        """识别成功因素"""
        factors = []
        
        # 分析消息历史寻找成功模式
        if self.current_round <= 3:
            factors.append("quick_convergence")
        
        review_messages = [m for m in self.message_history if m.get('type') == 'code_review']
        if len(review_messages) >= 1 and any('approved' in m.get('content', '').lower() for m in review_messages):
            factors.append("effective_code_review")
        
        compilation_messages = [m for m in self.message_history if m.get('type') == 'compilation_result']
        if compilation_messages and not any('failed' in m.get('content', '').lower() for m in compilation_messages[-2:]):
            factors.append("clean_compilation")
        
        return factors
    
    def _identify_failure_reasons(self) -> List[str]:
        """识别失败原因"""
        reasons = []
        
        if self.current_round >= self.max_rounds:
            reasons.append("max_rounds_exceeded")
        
        # 分析错误类型
        error_types = set()
        for message in self.message_history:
            content = message.get('content', '').lower()
            if 'compilation failed' in content:
                error_types.add("compilation_errors")
            elif 'simulation failed' in content:
                error_types.add("simulation_errors")
            elif 'needs improvement' in content:
                error_types.add("code_quality_issues")
        
        reasons.extend(list(error_types))
        
        return reasons
    
    def _suggest_improvements(self) -> List[str]:
        """建议改进"""
        suggestions = []
        
        if self.current_round >= self.max_rounds:
            suggestions.append("increase_max_rounds")
        
        error_count = sum(1 for m in self.message_history 
                         if 'error' in m.get('content', '').lower())
        if error_count > 3:
            suggestions.append("improve_error_handling")
            suggestions.append("add_more_detailed_feedback")
        
        if not self.test_code:
            suggestions.append("provide_test_code")
        
        return suggestions
    
    def _calculate_efficiency_score(self) -> float:
        """计算效率分数"""
        if self.success:
            # 成功的对话，轮次越少效率越高
            return max(0.1, 1.0 - (self.current_round - 1) / self.max_rounds)
        else:
            # 失败的对话，根据进展情况评分
            progress_indicators = 0
            
            # 是否有代码生成
            if any(m.get('type') == 'code_generation' for m in self.message_history):
                progress_indicators += 0.3
            
            # 是否通过编译
            if any('successful' in m.get('content', '').lower() and m.get('type') == 'compilation_result' 
                  for m in self.message_history):
                progress_indicators += 0.4
            
            # 是否有改进
            if len(set(m.get('sender') for m in self.message_history)) > 1:
                progress_indicators += 0.3
            
            return progress_indicators
    
    def _identify_bottlenecks(self) -> List[str]:
        """识别瓶颈"""
        bottlenecks = []
        
        # 分析各阶段耗时
        stage_counts = defaultdict(int)
        for message in self.message_history:
            msg_type = message.get('type', 'unknown')
            stage_counts[msg_type] += 1
        
        total_messages = len(self.message_history)
        
        # 如果某个阶段消息过多，可能是瓶颈
        for stage, count in stage_counts.items():
            if count / total_messages > 0.4:
                bottlenecks.append(f"excessive_{stage}")
        
        return bottlenecks
    
    async def _record_learning_experiences(self, result: Dict[str, Any]):
        """记录学习经验"""
        for participant in self.participants:
            if participant in self.framework.agents:
                agent = self.framework.agents[participant]
                
                experience = {
                    'type': 'dialogue_participation',
                    'dialogue_id': self.dialogue_id,
                    'success': result['success'],
                    'rounds': result['rounds'],
                    'role': agent.profile.role.value,
                    'final_code': result.get('final_code'),
                    'statistics': result['statistics'],
                    'timestamp': time.time()
                }
                
                # 添加代理商特定的经验数据
                if agent.profile.role == AgentRole.CODE_GENERATOR:
                    experience['code_generation_attempts'] = result['statistics']['action_counts'].get('code_generation', 0)
                elif agent.profile.role == AgentRole.CODE_REVIEWER:
                    experience['reviews_conducted'] = result['statistics']['action_counts'].get('code_review', 0)
                elif agent.profile.role == AgentRole.CODE_EXECUTOR:
                    experience['executions_performed'] = result['statistics']['action_counts'].get('code_execution', 0)
                
                # 使用高级学习系统处理经验
                learning_insights = await self.framework.learning_system.process_experience(
                    participant, experience
                )
                
                # 应用学习洞察到智能体
                await self._apply_learning_insights(agent, learning_insights)
                
                # 保存更新的智能体配置
                self.framework.persistence_manager.save_agent_profile(participant, agent.profile)
    
    async def _apply_learning_insights(self, agent: AutonomousAgent, insights: Dict[str, Any]):
        """应用学习洞察到智能体"""
        # 应用强化学习洞察
        rl_insights = insights.get('reinforcement_insights', {})
        if rl_insights.get('policy_insight'):
            policy = rl_insights['policy_insight']
            recommended_action = policy.get('recommended_action')
            
            # 根据推荐动作调整个性特征
            if recommended_action == 'creative':
                agent.profile.personality_traits[PersonalityTrait.CREATIVE] = min(1.0, 
                    agent.profile.personality_traits.get(PersonalityTrait.CREATIVE, 0.5) + 0.1)
            elif recommended_action == 'analytical':
                agent.profile.personality_traits[PersonalityTrait.ANALYTICAL] = min(1.0,
                    agent.profile.personality_traits.get(PersonalityTrait.ANALYTICAL, 0.5) + 0.1)
        
        # 应用模式识别洞察
        patterns = insights.get('patterns', [])
        for pattern in patterns:
            if pattern['type'] == 'success_pattern':
                # 增强成功相关的特征
                if 'quick_convergence' in pattern.get('pattern', {}):
                    agent.profile.personality_traits[PersonalityTrait.FOCUSED] = min(1.0,
                        agent.profile.personality_traits.get(PersonalityTrait.FOCUSED, 0.5) + 0.05)
            elif pattern['type'] == 'failure_pattern':
                # 调整失败相关的特征
                if 'compilation_error' in pattern.get('pattern', {}):
                    agent.profile.personality_traits[PersonalityTrait.DETAIL_ORIENTED] = min(1.0,
                        agent.profile.personality_traits.get(PersonalityTrait.DETAIL_ORIENTED, 0.5) + 0.1)
    
    async def _send_message_to_agent(self, agent: AutonomousAgent, message: Dict[str, Any]):
        """向智能体发送消息"""
        # 转换为UnifiedMessage格式
        unified_msg = UnifiedMessage(
            type=message.get('type', 'general'),
            content=message['content'],
            sender=message['sender'],
            receivers=[agent.profile.name],
            timestamp=message['timestamp'],
            conversation_id=self.dialogue_id
        )
        
        # 异步发送消息
        asyncio.create_task(agent.handle_message(unified_msg))
    
    def get_message_history(self) -> List[Dict[str, Any]]:
        """获取消息历史"""
        return self.message_history.copy()

# ============================================================================
# 6. HTML模板 (用于Web监控)
# ============================================================================

def create_dashboard_template():
    """创建监控面板HTML模板"""
    template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Auto-Dialogue Multi-Agent System Monitor</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; padding: 20px; background: #f5f5f5;
        }
        .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .card h3 { margin-top: 0; color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        .status { padding: 5px 10px; border-radius: 4px; color: white; font-weight: bold; }
        .status.running { background: #28a745; }
        .status.stopped { background: #dc3545; }
        .status.pending { background: #ffc107; color: #000; }
        .agent-list { list-style: none; padding: 0; }
        .agent-item { 
            display: flex; justify-content: space-between; align-items: center; 
            padding: 10px; margin: 5px 0; background: #f8f9fa; border-radius: 4px;
        }
        .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
        .metric { text-align: center; padding: 15px; background: #e9ecef; border-radius: 4px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
        .metric-label { font-size: 12px; color: #666; text-transform: uppercase; }
        .log-container { height: 300px; overflow-y: auto; border: 1px solid #dee2e6; padding: 10px; }
        .log-entry { padding: 5px; border-bottom: 1px solid #eee; font-family: monospace; font-size: 12px; }
        .task-form { display: grid; gap: 10px; }
        .task-form input, .task-form textarea, .task-form select { 
            padding: 8px; border: 1px solid #ddd; border-radius: 4px; 
        }
        .task-form button { 
            padding: 10px; background: #007bff; color: white; border: none; 
            border-radius: 4px; cursor: pointer; 
        }
        .task-form button:hover { background: #0056b3; }
        .dialogue-list { max-height: 300px; overflow-y: auto; }
        .dialogue-item { 
            padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; 
            background: #f8f9fa; border-radius: 0 4px 4px 0;
        }
    </style>
</head>
<body>
    <h1>🤖 Auto-Dialogue Multi-Agent System Monitor</h1>
    
    <div class="dashboard">
        <!-- System Status -->
        <div class="card">
            <h3>📊 System Status</h3>
            <div id="system-status">
                <p>Status: <span id="status-indicator" class="status">Loading...</span></p>
                <p>Uptime: <span id="uptime">-</span></p>
                <p>Active Agents: <span id="active-agents">-</span></p>
                <p>Running Dialogues: <span id="running-dialogues">-</span></p>
            </div>
        </div>
        
        <!-- Agents -->
        <div class="card">
            <h3>🤖 Agents</h3>
            <ul id="agent-list" class="agent-list"></ul>
        </div>
        
        <!-- Metrics -->
        <div class="card">
            <h3>📈 Metrics</h3>
            <div class="metrics" id="metrics-grid"></div>
        </div>
        
        <!-- Active Dialogues -->
        <div class="card">
            <h3>💬 Active Dialogues</h3>
            <div id="dialogue-list" class="dialogue-list"></div>
        </div>
        
        <!-- Submit New Task -->
        <div class="card">
            <h3>➕ Submit New Task</h3>
            <form class="task-form" id="task-form">
                <input type="text" id="task-description" placeholder="Problem description" required>
                <select id="task-type">
                    <option value="simple_logic">Simple Logic</option>
                    <option value="combinational">Combinational</option>
                    <option value="sequential">Sequential</option>
                    <option value="complex_module">Complex Module</option>
                </select>
                <textarea id="test-code" placeholder="Test code (optional)" rows="4"></textarea>
                <textarea id="custom-prompt" placeholder="Custom prompt (optional)" rows="2"></textarea>
                <input type="range" id="priority" min="0" max="1" step="0.1" value="0.5">
                <label for="priority">Priority: <span id="priority-value">0.5</span></label>
                <button type="submit">Submit Task</button>
            </form>
        </div>
        
        <!-- Live Events -->
        <div class="card">
            <h3>🔴 Live Events</h3>
            <div id="live-events" class="log-container"></div>
        </div>
    </div>

    <script>
        const socket = io();
        
        // Update priority display
        document.getElementById('priority').oninput = function() {
            document.getElementById('priority-value').textContent = this.value;
        };
        
        // Submit task form
        document.getElementById('task-form').onsubmit = async function(e) {
            e.preventDefault();
            
            const taskData = {
                description: document.getElementById('task-description').value,
                task_type: document.getElementById('task-type').value,
                test_code: document.getElementById('test-code').value,
                custom_prompt: document.getElementById('custom-prompt').value,
                priority: parseFloat(document.getElementById('priority').value)
            };
            
            try {
                const response = await fetch('/api/submit_task', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(taskData)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    addLiveEvent('Task submitted successfully: ' + result.task_id);
                    document.getElementById('task-form').reset();
                } else {
                    addLiveEvent('Task submission failed: ' + result.error);
                }
            } catch (error) {
                addLiveEvent('Error submitting task: ' + error.message);
            }
        };
        
        // Socket event handlers
        socket.on('connect', function() {
            addLiveEvent('Connected to monitoring system');
            requestUpdate();
        });
        
        socket.on('system_update', function(data) {
            updateSystemStatus(data);
        });
        
        socket.on('live_event', function(event) {
            addLiveEvent(`[${event.type}] ${JSON.stringify(event.data)}`);
        });
        
        function requestUpdate() {
            socket.emit('request_update');
        }
        
        function updateSystemStatus(data) {
            // Update status indicator
            const statusEl = document.getElementById('status-indicator');
            statusEl.textContent = data.running ? 'Running' : 'Stopped';
            statusEl.className = 'status ' + (data.running ? 'running' : 'stopped');
            
            // Update basic info
            document.getElementById('uptime').textContent = formatDuration(data.uptime || 0);
            document.getElementById('active-agents').textContent = Object.keys(data.agents || {}).length;
            document.getElementById('running-dialogues').textContent = data.active_dialogues || 0;
            
            // Update agents list
            updateAgentsList(data.agents || {});
            
            // Update metrics
            updateMetrics(data.system_metrics || {});
        }
        
        function updateAgentsList(agents) {
            const listEl = document.getElementById('agent-list');
            listEl.innerHTML = '';
            
            Object.entries(agents).forEach(([name, agent]) => {
                const li = document.createElement('li');
                li.className = 'agent-item';
                li.innerHTML = `
                    <div>
                        <strong>${name}</strong><br>
                        <small>${agent.role} | ${agent.personality}</small>
                    </div>
                    <div>
                        <div>Load: ${(agent.workload * 100).toFixed(0)}%</div>
                        <div>Convs: ${agent.active_conversations}</div>
                    </div>
                `;
                listEl.appendChild(li);
            });
        }
        
        function updateMetrics(metrics) {
            const gridEl = document.getElementById('metrics-grid');
            gridEl.innerHTML = '';
            
            const metricsToShow = [
                { key: 'completed_tasks', label: 'Tasks Done' },
                { key: 'success_rate', label: 'Success Rate', format: (v) => (v * 100).toFixed(1) + '%' },
                { key: 'average_task_duration', label: 'Avg Duration', format: (v) => formatDuration(v) },
                { key: 'total_errors', label: 'Total Errors' }
            ];
            
            metricsToShow.forEach(metric => {
                const value = metrics[metric.key] || 0;
                const formattedValue = metric.format ? metric.format(value) : value;
                
                const div = document.createElement('div');
                div.className = 'metric';
                div.innerHTML = `
                    <div class="metric-value">${formattedValue}</div>
                    <div class="metric-label">${metric.label}</div>
                `;
                gridEl.appendChild(div);
            });
        }
        
        function addLiveEvent(message) {
            const eventsEl = document.getElementById('live-events');
            const entryEl = document.createElement('div');
            entryEl.className = 'log-entry';
            entryEl.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            eventsEl.appendChild(entryEl);
            eventsEl.scrollTop = eventsEl.scrollHeight;
            
            // Keep only last 100 entries
            while (eventsEl.children.length > 100) {
                eventsEl.removeChild(eventsEl.firstChild);
            }
        }
        
        function formatDuration(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            
            if (hours > 0) {
                return `${hours}h ${minutes}m ${secs}s`;
            } else if (minutes > 0) {
                return `${minutes}m ${secs}s`;
            } else {
                return `${secs}s`;
            }
        }
        
        // Auto-refresh every 5 seconds
        setInterval(requestUpdate, 5000);
        
        // Initial load
        requestUpdate();
    </script>
</body>
</html>
"""
    return template

# ============================================================================
# 7. 主要使用示例和配置
# ============================================================================

async def main_example():
    """主要使用示例"""
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建配置文件
    config = {
        'llm': {
            'provider': 'openai',
            'base_url': 'https://api.openai.com/v1',
            'api_key': 'your-api-key-here',
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 2000
        },
        'dialogue': {
            'max_rounds': 10,
            'timeout_minutes': 30
        }
    }
    
    # 保存配置
    config_path = 'auto_dialogue_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # 创建框架
    framework = AutoDialogueFramework(config_path, max_dialogue_rounds=10)
    
    try:
        # 初始化
        await framework.initialize()
        
        # 启动系统
        await framework.start()
        
        print("🚀 Auto-Dialogue Framework started!")
        print("📊 Web Monitor: http://localhost:5000")
        print("🤖 Agents ready for autonomous dialogue")
        
        # 示例任务1：简单逻辑设计
        print("\n=== Example 1: Simple Logic Design ===")
        
        problem1 = "Design a 4-bit counter with synchronous reset and enable"
        test_code1 = """
module counter_tb;
    reg clk, reset, enable;
    wire [3:0] count;
    
    counter dut(.clk(clk), .reset(reset), .enable(enable), .count(count));
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $dumpfile("counter.vcd");
        $dumpvars(0, counter_tb);
        
        reset = 1; enable = 0;
        #20 reset = 0;
        #10 enable = 1;
        #100 enable = 0;
        #20 enable = 1;
        #50 $finish;
    end
endmodule
"""
        
        dialogue_id1 = await framework.start_auto_dialogue(
            problem_description=problem1,
            test_code=test_code1,
            custom_prompt="Focus on clean, readable code with proper reset behavior"
        )
        
        print(f"✅ Dialogue {dialogue_id1} completed")
        
        # 示例任务2：更复杂的设计
        print("\n=== Example 2: Complex Module Design ===")
        
        problem2 = "Design a simple FIFO buffer (4 entries, 8-bit wide) with full/empty flags"
        
        dialogue_id2 = await framework.start_auto_dialogue(
            problem_description=problem2,
            custom_prompt="Include comprehensive error checking and edge case handling"
        )
        
        print(f"✅ Dialogue {dialogue_id2} completed")
        
        # 等待用户输入或继续运行
        print("\n🔄 System running... Press Ctrl+C to stop")
        print("💡 You can submit new tasks via the web interface")
        
        # 保持运行状态
        while True:
            await asyncio.sleep(10)
            
            # 显示系统状态
            status = framework.get_system_status()
            print(f"📊 Active dialogues: {len(framework.active_dialogues)}")
            print(f"🤖 Agents: {len(framework.agents)}")
            print(f"💾 Tasks in history: {len(framework.dialogue_history)}")
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # 清理资源
        framework.toolchain.cleanup()
        await framework.shutdown()

# ============================================================================
# 8. 辅助工具和实用函数
# ============================================================================

def create_templates_directory():
    """创建HTML模板目录"""
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # 保存dashboard模板
    dashboard_path = templates_dir / "dashboard.html"
    with open(dashboard_path, 'w') as f:
        f.write(create_dashboard_template())
    
    print(f"✅ Templates created in {templates_dir}")

def create_example_configs():
    """创建示例配置文件"""
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # 基础配置
    basic_config = {
        'llm': {
            'provider': 'openai',
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 2000,
            'api_key': 'your-api-key-here'
        },
        'dialogue': {
            'max_rounds': 8,
            'timeout_minutes': 20,
            'auto_escalation': True
        },
        'agents': {
            'coder': {
                'personality': {
                    'creative': 0.8,
                    'proactive': 0.7,
                    'focused': 0.6
                },
                'capabilities': ['code_generation', 'debugging', 'optimization']
            },
            'reviewer': {
                'personality': {
                    'analytical': 0.9,
                    'detail_oriented': 0.9,
                    'cautious': 0.8
                },
                'capabilities': ['code_review', 'error_analysis', 'quality_assurance']
            },
            'executor': {
                'personality': {
                    'focused': 0.9,
                    'reliable': 0.9,
                    'proactive': 0.6
                },
                'capabilities': ['compilation', 'simulation', 'testing']
            }
        },
        'toolchain': {
            'work_dir': './verilog_workspace',
            'cleanup_after_task': True,
            'simulation_timeout': 30
        },
        'monitoring': {
            'web_port': 5000,
            'enable_real_time_updates': True,
            'log_level': 'INFO'
        }
    }
    
    # 高级配置（更多轮次，复杂任务）
    advanced_config = basic_config.copy()
    advanced_config.update({
        'dialogue': {
            'max_rounds': 15,
            'timeout_minutes': 45,
            'auto_escalation': True,
            'intervention_threshold': 4
        },
        'learning': {
            'enable_reinforcement_learning': True,
            'pattern_recognition': True,
            'knowledge_transfer': True,
            'collective_intelligence': True,
            'learning_rate': 0.1
        }
    })
    
    # 快速测试配置
    test_config = {
        'llm': {
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'temperature': 0.5,
            'max_tokens': 1000,
            'api_key': 'your-api-key-here'
        },
        'dialogue': {
            'max_rounds': 5,
            'timeout_minutes': 10
        },
        'toolchain': {
            'work_dir': './test_workspace',
            'cleanup_after_task': True
        }
    }
    
    configs = {
        'basic_auto_dialogue.yaml': basic_config,
        'advanced_auto_dialogue.yaml': advanced_config,
        'test_auto_dialogue.yaml': test_config
    }
    
    for filename, config in configs.items():
        config_path = configs_dir / filename
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Example configs created in {configs_dir}")

class TaskTemplates:
    """预定义任务模板"""
    
    @staticmethod
    def get_simple_logic_tasks():
        """简单逻辑任务模板"""
        return [
            {
                'name': '2-to-1 Multiplexer',
                'description': 'Design a 2-to-1 multiplexer with 4-bit inputs',
                'test_code': '''
module mux_tb;
    reg [3:0] a, b;
    reg sel;
    wire [3:0] out;
    
    mux2to1 dut(.a(a), .b(b), .sel(sel), .out(out));
    
    initial begin
        $dumpfile("mux.vcd");
        $dumpvars(0, mux_tb);
        
        a = 4'b1010; b = 4'b0101; sel = 0;
        #10 sel = 1;
        #10 a = 4'b1111; b = 4'b0000;
        #10 sel = 0;
        #10 $finish;
    end
endmodule
'''
            },
            {
                'name': '4-bit Adder',
                'description': 'Design a 4-bit binary adder with carry output',
                'test_code': '''
module adder_tb;
    reg [3:0] a, b;
    reg cin;
    wire [3:0] sum;
    wire cout;
    
    adder4bit dut(.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));
    
    initial begin
        $dumpfile("adder.vcd");
        $dumpvars(0, adder_tb);
        
        cin = 0;
        a = 4'b0000; b = 4'b0000; #10;
        a = 4'b0101; b = 4'b0011; #10;
        a = 4'b1111; b = 4'b0001; #10;
        a = 4'b1111; b = 4'b1111; #10;
        $finish;
    end
endmodule
'''
            }
        ]
    
    @staticmethod
    def get_sequential_logic_tasks():
        """时序逻辑任务模板"""
        return [
            {
                'name': 'D Flip-Flop',
                'description': 'Design a D flip-flop with asynchronous reset',
                'test_code': '''
module dff_tb;
    reg clk, reset, d;
    wire q, qn;
    
    dff dut(.clk(clk), .reset(reset), .d(d), .q(q), .qn(qn));
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $dumpfile("dff.vcd");
        $dumpvars(0, dff_tb);
        
        reset = 1; d = 0;
        #15 reset = 0;
        #10 d = 1;
        #20 d = 0;
        #20 d = 1;
        #15 reset = 1;
        #10 reset = 0;
        #20 $finish;
    end
endmodule
'''
            },
            {
                'name': 'Shift Register',
                'description': 'Design an 8-bit shift register with parallel load',
                'test_code': '''
module shiftreg_tb;
    reg clk, reset, load, shift_en;
    reg [7:0] parallel_in;
    reg serial_in;
    wire [7:0] parallel_out;
    wire serial_out;
    
    shift_register dut(
        .clk(clk), .reset(reset), .load(load), 
        .shift_en(shift_en), .parallel_in(parallel_in), 
        .serial_in(serial_in), .parallel_out(parallel_out), 
        .serial_out(serial_out)
    );
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $dumpfile("shiftreg.vcd");
        $dumpvars(0, shiftreg_tb);
        
        reset = 1; load = 0; shift_en = 0;
        parallel_in = 8'b10110010; serial_in = 0;
        #15 reset = 0;
        #10 load = 1;
        #10 load = 0; shift_en = 1; serial_in = 1;
        #80 shift_en = 0;
        #20 $finish;
    end
endmodule
'''
            }
        ]

def create_setup_script():
    """创建安装脚本"""
    setup_script = '''#!/bin/bash

echo "🚀 Setting up Auto-Dialogue Multi-Agent System..."

# Create directory structure
mkdir -p {data,logs,verilog_workspace,templates,configs,test_results}

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Check for Verilog tools
echo "🔧 Checking Verilog tools..."

if ! command -v iverilog &> /dev/null; then
    echo "⚠️  Icarus Verilog not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get install -y iverilog
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install icarus-verilog
    else
        echo "❌ Please install Icarus Verilog manually"
    fi
else
    echo "✅ Icarus Verilog found"
fi

if ! command -v gtkwave &> /dev/null; then
    echo "⚠️  GTKWave not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install -y gtkwave
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install gtkwave
    else
        echo "❌ Please install GTKWave manually"
    fi
else
    echo "✅ GTKWave found"
fi

if ! command -v verilator &> /dev/null; then
    echo "⚠️  Verilator not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install -y verilator
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install verilator
    else
        echo "❌ Please install Verilator manually"
    fi
else
    echo "✅ Verilator found"
fi

# Create initial configuration
echo "⚙️  Creating initial configuration..."
python3 -c "
from auto_dialogue_framework import create_example_configs, create_templates_directory
create_example_configs()
create_templates_directory()
print('✅ Configuration files created')
"

echo "🎉 Setup complete!"
echo ""
echo "📖 Next steps:"
echo "1. Edit configs/basic_auto_dialogue.yaml with your LLM API key"
echo "2. Run: python auto_dialogue_example.py"
echo "3. Open web monitor: http://localhost:5000"
echo ""
echo "📚 For more information, see the documentation in docs/"
'''
    
    with open('setup.sh', 'w') as f:
        f.write(setup_script)
    
    # Make executable
    os.chmod('setup.sh', 0o755)
    print("✅ Setup script created: setup.sh")

def create_requirements_file():
    """创建requirements.txt"""
    requirements = '''
# Core dependencies
asyncio
pyyaml
dataclasses
pathlib
sqlite3
pickle
subprocess
tempfile
shutil
threading
collections
enum

# LLM integration
openai>=1.0.0
anthropic
requests

# Web interface
flask>=2.0.0
flask-socketio>=5.0.0

# Data processing
numpy
pandas

# Logging and monitoring
coloredlogs
psutil

# Testing
pytest
pytest-asyncio

# Optional: Advanced features
torch  # For reinforcement learning
scikit-learn  # For pattern recognition
networkx  # For graph analysis
plotly  # For advanced visualization
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())
    
    print("✅ Requirements file created: requirements.txt")

# ============================================================================
# 9. 主运行脚本
# ============================================================================

def create_main_runner():
    """创建主运行脚本"""
    runner_script = '''#!/usr/bin/env python3
"""
Auto-Dialogue Multi-Agent System - Main Runner
Run this script to start the complete system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from auto_dialogue_framework import (
    AutoDialogueFramework, 
    create_example_configs, 
    create_templates_directory,
    TaskTemplates
)

async def interactive_mode(framework):
    """交互模式"""
    print("\\n🤖 Interactive Mode - Submit tasks interactively")
    print("Commands: 'task', 'status', 'quit', 'help'")
    
    while True:
        try:
            cmd = input("\\n> ").strip().lower()
            
            if cmd == 'quit' or cmd == 'exit':
                break
            
            elif cmd == 'help':
                print("""
Available commands:
- task: Submit a new task
- status: Show system status  
- templates: Show available task templates
- agents: Show agent information
- history: Show dialogue history
- quit: Exit interactive mode
""")
            
            elif cmd == 'task':
                await submit_interactive_task(framework)
            
            elif cmd == 'status':
                show_system_status(framework)
            
            elif cmd == 'templates':
                show_task_templates()
            
            elif cmd == 'agents':
                show_agents_info(framework)
            
            elif cmd == 'history':
                show_dialogue_history(framework)
            
            else:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

async def submit_interactive_task(framework):
    """交互式提交任务"""
    print("\\n📝 Task Submission")
    
    # Get task description
    description = input("Problem description: ").strip()
    if not description:
        print("❌ Description is required")
        return
    
    # Optional test code
    print("Test code (optional, press Enter twice to finish):")
    test_lines = []
    while True:
        line = input()
        if line == "" and (not test_lines or test_lines[-1] == ""):
            break
        test_lines.append(line)
    
    test_code = "\\n".join(test_lines).strip() if test_lines else None
    
    # Optional custom prompt
    custom_prompt = input("Custom prompt (optional): ").strip() or None
    
    print("\\n🚀 Starting dialogue...")
    
    try:
        dialogue_id = await framework.start_auto_dialogue(
            problem_description=description,
            test_code=test_code,
            custom_prompt=custom_prompt
        )
        
        print(f"✅ Dialogue {dialogue_id} completed successfully!")
        
    except Exception as e:
        print(f"❌ Dialogue failed: {e}")

def show_system_status(framework):
    """显示系统状态"""
    status = framework.get_system_status()
    
    print("\\n📊 System Status:")
    print(f"Running: {'✅' if status.get('running', False) else '❌'}")
    print(f"Active Agents: {len(status.get('agents', {}))}")
    print(f"Active Dialogues: {len(framework.active_dialogues)}")
    print(f"Completed Dialogues: {len(framework.dialogue_history)}")
    
    if framework.active_dialogues:
        print("\\n💬 Active Dialogues:")
        for dialogue_id, dialogue in framework.active_dialogues.items():
            print(f"  - {dialogue_id}: Round {dialogue.current_round}/{dialogue.max_rounds}")

def show_task_templates():
    """显示任务模板"""
    print("\\n📋 Available Task Templates:")
    
    print("\\n🔹 Simple Logic:")
    for i, task in enumerate(TaskTemplates.get_simple_logic_tasks(), 1):
        print(f"  {i}. {task['name']}")
    
    print("\\n🔸 Sequential Logic:")
    for i, task in enumerate(TaskTemplates.get_sequential_logic_tasks(), 1):
        print(f"  {i}. {task['name']}")

def show_agents_info(framework):
    """显示智能体信息"""
    print("\\n🤖 Agents Information:")
    
    for name, agent in framework.agents.items():
        print(f"\\n  {name}:")
        print(f"    Role: {agent.profile.role.value}")
        print(f"    Workload: {agent.profile.current_workload:.1%}")
        print(f"    Success Rate: {agent.profile.success_rate:.1%}")
        print(f"    Active Conversations: {len(agent.active_conversations)}")

def show_dialogue_history(framework):
    """显示对话历史"""
    print("\\n📚 Dialogue History:")
    
    if not framework.dialogue_history:
        print("  No completed dialogues yet.")
        return
    
    for dialogue in list(framework.dialogue_history)[-10:]:  # Show last 10
        success_icon = "✅" if dialogue.success else "❌"
        print(f"  {success_icon} {dialogue.dialogue_id}: "
              f"{dialogue.current_round} rounds, "
              f"{'completed' if dialogue.task_completed else 'incomplete'}")

async def run_demo_tasks(framework):
    """运行演示任务"""
    print("\\n🎯 Running Demo Tasks...")
    
    # Demo 1: Simple multiplexer
    print("\\n=== Demo 1: 2-to-1 Multiplexer ===")
    simple_tasks = TaskTemplates.get_simple_logic_tasks()
    mux_task = simple_tasks[0]
    
    dialogue_id = await framework.start_auto_dialogue(
        problem_description=mux_task['description'],
        test_code=mux_task['test_code']
    )
    print(f"✅ Demo 1 completed: {dialogue_id}")
    
    # Demo 2: D Flip-Flop
    print("\\n=== Demo 2: D Flip-Flop ===")
    sequential_tasks = TaskTemplates.get_sequential_logic_tasks()
    dff_task = sequential_tasks[0]
    
    dialogue_id = await framework.start_auto_dialogue(
        problem_description=dff_task['description'],
        test_code=dff_task['test_code']
    )
    print(f"✅ Demo 2 completed: {dialogue_id}")

async def main():
    """主函数"""
    print("🚀 Auto-Dialogue Multi-Agent System")
    print("=====================================")
    
    # Check if config exists
    config_path = "configs/basic_auto_dialogue.yaml"
    if not os.path.exists(config_path):
        print("⚙️  Creating initial configuration...")
        create_example_configs()
        create_templates_directory()
        print(f"✅ Please edit {config_path} with your LLM API key")
        return
    
    # Create framework
    framework = AutoDialogueFramework(config_path)
    
    try:
        # Initialize
        await framework.initialize()
        await framework.start()
        
        print("✅ System initialized successfully!")
        print(f"📊 Web Monitor: http://localhost:5000")
        
        # Parse command line arguments
        if len(sys.argv) > 1:
            mode = sys.argv[1]
            
            if mode == "demo":
                await run_demo_tasks(framework)
            elif mode == "interactive":
                await interactive_mode(framework)
            elif mode == "server":
                print("🔄 Running in server mode...")
                print("💡 Submit tasks via web interface or API")
                while True:
                    await asyncio.sleep(10)
            else:
                print(f"Unknown mode: {mode}")
                print("Available modes: demo, interactive, server")
        else:
            # Default: interactive mode
            await interactive_mode(framework)
    
    except KeyboardInterrupt:
        print("\\n🛑 Shutting down...")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        framework.toolchain.cleanup()
        await framework.shutdown()
        print("✅ Shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open('run_auto_dialogue.py', 'w') as f:
        f.write(runner_script)
    
    # Make executable
    os.chmod('run_auto_dialogue.py', 0o755)
    print("✅ Main runner created: run_auto_dialogue.py")

# ============================================================================
# 10. 完整的系统状态和工具函数
# ============================================================================

class SystemStatusManager:
    """系统状态管理器"""
    
    def __init__(self, framework: AutoDialogueFramework):
        self.framework = framework
        self.start_time = time.time()
        self.logger = setup_logger("SystemStatus")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """获取完整的系统状态"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # 基础状态
        status = {
            'running': True,
            'uptime': uptime,
            'start_time': self.start_time,
            'current_time': current_time
        }
        
        # 智能体状态
        agents_status = {}
        for name, agent in self.framework.agents.items():
            agents_status[name] = {
                'name': agent.profile.name,
                'role': agent.profile.role.value,
                'workload': agent.profile.current_workload,
                'success_rate': agent.profile.success_rate,
                'active_conversations': len(agent.active_conversations),
                'personality': {
                    trait.value: value 
                    for trait, value in agent.profile.personality_traits.items()
                },
                'capabilities': [cap.value for cap in agent.profile.capabilities.keys()],
                'last_activity': getattr(agent, 'last_activity_time', current_time)
            }
        
        status['agents'] = agents_status
        
        # 对话状态
        status['active_dialogues'] = len(self.framework.active_dialogues)
        status['completed_dialogues'] = len(self.framework.dialogue_history)
        
        # 详细对话信息
        status['dialogue_details'] = {
            'active': [
                {
                    'id': dialogue.dialogue_id,
                    'round': dialogue.current_round,
                    'max_rounds': dialogue.max_rounds,
                    'participants': dialogue.participants,
                    'problem': dialogue.problem_description[:100] + "..." if len(dialogue.problem_description) > 100 else dialogue.problem_description
                }
                for dialogue in self.framework.active_dialogues.values()
            ],
            'recent_completed': [
                {
                    'id': dialogue.dialogue_id,
                    'success': dialogue.success,
                    'rounds': dialogue.current_round,
                    'participants': dialogue.participants
                }
                for dialogue in list(self.framework.dialogue_history)[-5:]
            ]
        }
        
        # 系统资源
        status['system_resources'] = self._get_system_resources()
        
        # 性能指标
        status['performance_metrics'] = self._get_performance_metrics()
        
        return status
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """获取系统资源信息"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('.').percent if hasattr(psutil, 'disk_usage') else 0,
                'process_count': len(psutil.pids()),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except ImportError:
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'disk_usage': 0,
                'process_count': 0,
                'load_average': [0, 0, 0]
            }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        completed_dialogues = list(self.framework.dialogue_history)
        
        if not completed_dialogues:
            return {
                'total_dialogues': 0,
                'success_rate': 0.0,
                'average_rounds': 0.0,
                'average_duration': 0.0,
                'error_rate': 0.0
            }
        
        successful = sum(1 for d in completed_dialogues if d.success)
        total_rounds = sum(d.current_round for d in completed_dialogues)
        
        # 计算平均持续时间（如果有记录的话）
        total_duration = 0
        for dialogue in completed_dialogues:
            if hasattr(dialogue, 'start_time') and hasattr(dialogue, 'end_time'):
                total_duration += dialogue.end_time - dialogue.start_time
        
        return {
            'total_dialogues': len(completed_dialogues),
            'success_rate': successful / len(completed_dialogues) if completed_dialogues else 0.0,
            'average_rounds': total_rounds / len(completed_dialogues) if completed_dialogues else 0.0,
            'average_duration': total_duration / len(completed_dialogues) if completed_dialogues and total_duration > 0 else 0.0,
            'error_rate': (len(completed_dialogues) - successful) / len(completed_dialogues) if completed_dialogues else 0.0
        }

# 将SystemStatusManager集成到主框架
def integrate_status_manager(framework_class):
    """将状态管理器集成到框架中"""
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态（集成版本）"""
        if not hasattr(self, '_status_manager'):
            self._status_manager = SystemStatusManager(self)
        
        return self._status_manager.get_comprehensive_status()
    
    # 添加方法到框架类
    framework_class.get_system_status = get_system_status
    
    return framework_class

# 应用集成
AutoDialogueFramework = integrate_status_manager(AutoDialogueFramework)

# ============================================================================
# 11. 最终的完整示例和入口点
# ============================================================================

if __name__ == "__main__":
    """
    如果直接运行此文件，执行完整的设置和演示
    """
    
    print("🚀 Auto-Dialogue Multi-Agent System Setup")
    print("==========================================")
    
    # 创建所有必要的文件和配置
    print("📁 Creating project structure...")
    create_example_configs()
    create_templates_directory()
    create_requirements_file()
    create_setup_script()
    create_main_runner()
    
    print("\n✅ Setup complete!")
    print("\n📖 Quick start guide:")
    print("1. Run: chmod +x setup.sh && ./setup.sh")
    print("2. Edit configs/basic_auto_dialogue.yaml with your LLM API key")
    print("3. Run: python run_auto_dialogue.py interactive")
    print("4. Open web monitor: http://localhost:5000")
    print("\n🎯 Or run a demo: python run_auto_dialogue.py demo")
    print("🖥️  Or run as server: python run_auto_dialogue.py server")
    
    print("\n🔧 Available modes:")
    print("- interactive: Command-line interface for submitting tasks")
    print("- demo: Run predefined demonstration tasks")
    print("- server: Run in background, use web interface")
    
    print("\n📚 For detailed documentation, see the docs/ directory")
    print("🐛 For issues and contributions, check the project repository")
                    