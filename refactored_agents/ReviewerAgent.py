from unified_autonomous_framework import *
import asyncio
import logging
from typing import Dict, Any, Optional, List, Set

class ReviewerAgent(AutonomousAgent):
    """重构后的代码审查智能体"""
    
    def __init__(self, name: str = "ReviewerAgent", system_coordinator: SystemCoordinator = None):
        # 创建Reviewer专属的Profile
        profile = UnifiedAgentProfile(
            name=name,
            role=AgentRole.CODE_REVIEWER,
            communication_style=CommunicationStyle.ANALYTICAL,
            personality_traits={
                PersonalityTrait.ANALYTICAL: 0.95,
                PersonalityTrait.DETAIL_ORIENTED: 0.9,
                PersonalityTrait.HELPFUL: 0.8,
                PersonalityTrait.CRITICAL: 0.85,
                PersonalityTrait.RELIABLE: 0.9
            },
            capabilities={
                AgentCapability.CODE_REVIEW: 0.95,
                AgentCapability.ERROR_ANALYSIS: 0.9,
                AgentCapability.CODE_OPTIMIZATION: 0.8,
                AgentCapability.VERIFICATION: 0.85,
                AgentCapability.DOCUMENTATION: 0.7
            },
            proactivity=0.7,
            learning_rate=0.15
        )
        
        # 初始化父类
        super().__init__(profile, system_coordinator)
        
        # Reviewer特有的组件
        self.code_analyzer = VerilogCodeAnalyzer(self)
        self.review_patterns = self._load_review_patterns()
        self.review_history = deque(maxlen=100)
        self.common_issues_db = self._build_common_issues_database()
        
        self.logger = logging.getLogger(f"ReviewerAgent.{name}")
    
    def _load_review_patterns(self) -> Dict[str, List[str]]:
        """加载审查模式"""
        return {
            "synthesis": [
                "Check for latches in combinational logic",
                "Verify complete case statements",
                "Check for multiple drivers"
            ],
            "timing": [
                "Check clock domain crossings",
                "Verify setup and hold requirements",
                "Check for combinational loops"
            ],
            "style": [
                "Naming conventions",
                "Code formatting",
                "Comment quality"
            ],
            "functionality": [
                "Reset behavior",
                "Edge cases handling",
                "State machine completeness"
            ]
        }
    
    def _build_common_issues_database(self) -> Dict[str, Dict[str, Any]]:
        """构建常见问题数据库"""
        return {
            "incomplete_sensitivity_list": {
                "severity": "HIGH",
                "pattern": r"always\s*@\s*\([^)]*\)",
                "fix_suggestion": "Include all read signals in sensitivity list"
            },
            "unintended_latch": {
                "severity": "HIGH",
                "pattern": r"always\s*@\s*\*.*if.*(?!else)",
                "fix_suggestion": "Add else clause to prevent latch inference"
            },
            "missing_default_case": {
                "severity": "MEDIUM",
                "pattern": r"case\s*\([^)]*\)(?!.*default)",
                "fix_suggestion": "Add default case to handle all possibilities"
            }
        }
    
    async def _perception_loop(self):
        """Reviewer的感知循环"""
        while self.behavior_loop_active:
            try:
                # 感知待审查的代码
                pending_reviews = await self._scan_for_review_requests()
                
                # 更新决策上下文
                self.decision_context["pending_reviews"] = pending_reviews
                self.decision_context["review_opportunities"] = len(pending_reviews)
                
                # 感知质量问题趋势
                quality_trends = self._analyze_quality_trends()
                if quality_trends["declining_quality"]:
                    self.decision_context["quality_alerts"].append({
                        "type": "quality_decline",
                        "areas": quality_trends["problem_areas"],
                        "suggested_action": "proactive_guidance"
                    })
                
                # 感知协作机会
                if pending_reviews:
                    for review in pending_reviews[:5]:
                        if review.get("complexity", 0) > 0.8:
                            self.decision_context["collaboration_opportunities"].append({
                                "task": review,
                                "reason": "complex_review",
                                "suggested_partners": ["ExecutorAgent"]
                            })
                
                await asyncio.sleep(3)
                
            except Exception as e:
                self.logger.error(f"Error in Reviewer perception loop: {e}")
                await asyncio.sleep(3)
    
    async def _proactive_behavior_loop(self):
        """Reviewer的主动行为循环"""
        while self.behavior_loop_active:
            try:
                # 主动提供代码质量建议
                if self.profile.proactivity > 0.6:
                    # 检查最近的代码质量趋势
                    if self._should_provide_proactive_guidance():
                        await self._provide_quality_guidance()
                
                # 主动分享审查经验
                if len(self.review_history) > 20:
                    insights = self._extract_review_insights()
                    if insights:
                        await self._share_review_insights(insights)
                
                await asyncio.sleep(120)  # 2分钟检查一次
                
            except Exception as e:
                self.logger.error(f"Error in proactive behavior loop: {e}")
                await asyncio.sleep(120)
    
    async def _scan_for_review_requests(self) -> List[Dict[str, Any]]:
        """扫描待审查的请求"""
        pending_reviews = []
        
        # 从决策上下文中查找
        for msg_info in self.decision_context.get("pending_messages", []):
            message = msg_info["message"]
            if message.intent == MessageIntent.TASK_RESULT:
                content = message.content
                if content.get("ready_for_review", False):
                    pending_reviews.append({
                        "task_id": content.get("task_id"),
                        "code": content.get("artifacts", {}).get("verilog_code"),
                        "sender": message.sender,
                        "complexity": self._estimate_review_complexity(content),
                        "priority": message.priority
                    })
        
        return pending_reviews
    
    def _estimate_review_complexity(self, content: Dict[str, Any]) -> float:
        """估算审查复杂度"""
        code = content.get("artifacts", {}).get("verilog_code", "")
        
        complexity = 0.3  # 基础复杂度
        
        # 基于代码特征调整
        if "always @" in code:
            complexity += 0.2
        if "case" in code or "if" in code:
            complexity += 0.1
        if len(code.split('\n')) > 50:
            complexity += 0.2
        
        return min(1.0, complexity)
    
    async def handle_message(self, message: UnifiedMessage) -> Optional[UnifiedMessage]:
        """处理消息 - 重写以支持审查特定逻辑"""
        # 特殊处理审查请求
        if message.intent == MessageIntent.TASK_RESULT and message.content.get("ready_for_review"):
            # 立即开始审查
            await self._start_review_process(message)
            return None
        
        # 其他消息使用标准处理
        return await super().handle_message(message)
    
    async def _start_review_process(self, message: UnifiedMessage):
        """开始审查流程"""
        content = message.content
        task_id = content.get("task_id")
        code = content.get("artifacts", {}).get("verilog_code")
        
        if not code:
            self.logger.warning(f"No code found for review in task {task_id}")
            return
        
        self.logger.info(f"Starting review for task {task_id}")
        
        # 执行审查
        review_result = await self.code_analyzer.analyze(code, content.get("design_approach"))
        
        # 生成审查报告
        report = self._generate_review_report(task_id, review_result)
        
        # 记录到历史
        self.review_history.append({
            "task_id": task_id,
            "issues_found": len(review_result["issues"]),
            "severity_distribution": review_result["severity_distribution"],
            "timestamp": time.time(),
            "reviewed_agent": message.sender
        })
        
        # 准备审查结果消息
        review_message = UnifiedMessage(
            type="review_result",
            content={
                "task_id": task_id,
                "review_status": review_result["status"],
                "issues": review_result["issues"],
                "suggestions": review_result["suggestions"],
                "score": review_result["score"],
                "detailed_report": report,
                "reviewer": self.profile.name
            },
            sender=self.profile.name,
            receivers=[message.sender, "ExecutorAgent", "SystemCoordinator"],
            priority=MessagePriority.HIGH if review_result["status"] == "rejected" else MessagePriority.NORMAL,
            intent=MessageIntent.REVIEW_RESULT
        )
        
        # 发送审查结果
        await self.message_router.route_message(review_message, self.profile.name)
        
        # 学习记录
        self.experience_buffer.append({
            "task_type": "code_review",
            "complexity": self._estimate_review_complexity(content),
            "issues_found": len(review_result["issues"]),
            "success": True,
            "duration": time.time() - message.timestamp if hasattr(message, 'timestamp') else 0
        })
    
    def _generate_review_report(self, task_id: str, review_result: Dict[str, Any]) -> str:
        """生成审查报告"""
        report_parts = [
            f"# Code Review Report for Task {task_id}",
            f"Reviewer: {self.profile.name}",
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Summary",
            f"- Overall Score: {review_result['score']}/100",
            f"- Status: {review_result['status']}",
            f"- Total Issues: {len(review_result['issues'])}",
            f"\n## Issues Found"
        ]
        
        # 按严重程度分组
        for severity in ["HIGH", "MEDIUM", "LOW"]:
            severity_issues = [i for i in review_result["issues"] if i["severity"] == severity]
            if severity_issues:
                report_parts.append(f"\n### {severity} Severity Issues")
                for issue in severity_issues:
                    report_parts.append(f"- **{issue['type']}**: {issue['description']}")
                    report_parts.append(f"  - Location: Line {issue.get('line', 'N/A')}")
                    report_parts.append(f"  - Suggestion: {issue['suggestion']}")
        
        # 添加建议
        if review_result["suggestions"]:
            report_parts.append(f"\n## Recommendations")
            for suggestion in review_result["suggestions"]:
                report_parts.append(f"- {suggestion}")
        
        return "\n".join(report_parts)
    
    def _analyze_quality_trends(self) -> Dict[str, Any]:
        """分析代码质量趋势"""
        if len(self.review_history) < 10:
            return {"declining_quality": False, "problem_areas": []}
        
        recent_reviews = list(self.review_history)[-10:]
        
        # 计算平均问题数
        avg_issues = sum(r["issues_found"] for r in recent_reviews) / len(recent_reviews)
        
        # 检查是否质量下降
        first_half_avg = sum(r["issues_found"] for r in recent_reviews[:5]) / 5
        second_half_avg = sum(r["issues_found"] for r in recent_reviews[5:]) / 5
        
        declining = second_half_avg > first_half_avg * 1.5
        
        # 识别问题领域
        problem_areas = []
        if declining:
            # 分析最常见的问题类型
            issue_types = defaultdict(int)
            for review in recent_reviews[5:]:
                for issue in review.get("issues", []):
                    issue_types[issue.get("type", "unknown")] += 1
            
            problem_areas = [k for k, v in issue_types.items() if v > 3]
        
        return {
            "declining_quality": declining,
            "problem_areas": problem_areas,
            "average_issues": avg_issues
        }
    
    def _should_provide_proactive_guidance(self) -> bool:
        """决定是否应该主动提供指导"""
        # 基于个性和质量趋势
        if self.profile.personality_traits.get(PersonalityTrait.HELPFUL, 0.5) > 0.7:
            trends = self._analyze_quality_trends()
            return trends["declining_quality"] or trends["average_issues"] > 5
        
        return False
    
    async def _provide_quality_guidance(self):
        """主动提供质量指导"""
        trends = self._analyze_quality_trends()
        
        guidance_message = UnifiedMessage(
            type="quality_guidance",
            content={
                "type": "proactive_quality_tips",
                "problem_areas": trends["problem_areas"],
                "tips": self._generate_quality_tips(trends["problem_areas"]),
                "reviewer": self.profile.name
            },
            sender=self.profile.name,
            receivers=["CoderAgent"],  # 发给所有Coder
            priority=MessagePriority.LOW,
            intent=MessageIntent.INFORMATION_SHARING
        )
        
        await self.message_router.route_message(guidance_message, self.profile.name)
    
    def _generate_quality_tips(self, problem_areas: List[str]) -> List[str]:
        """生成质量提示"""
        tips = []
        
        tip_database = {
            "incomplete_sensitivity_list": "Always include all signals read in combinational always blocks",
            "unintended_latch": "Ensure all if statements have else clauses in combinational logic",
            "missing_default_case": "Add default cases to handle unexpected inputs"
        }
        
        for area in problem_areas:
            if area in tip_database:
                tips.append(tip_database[area])
        
        # 添加通用建议
        if not tips:
            tips.append("Consider code review checklist before submission")
            tips.append("Run synthesis to catch common issues early")
        
        return tips


class VerilogCodeAnalyzer:
    """Verilog代码分析器"""
    
    def __init__(self, agent: ReviewerAgent):
        self.agent = agent
        self.logger = logging.getLogger(f"CodeAnalyzer.{agent.profile.name}")
    
    async def analyze(self, code: str, design_approach: str = None) -> Dict[str, Any]:
        """分析Verilog代码"""
        result = {
            "status": "pending",
            "issues": [],
            "suggestions": [],
            "score": 100,
            "severity_distribution": {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        }
        
        # 执行多种分析
        syntax_issues = await self._check_syntax(code)
        style_issues = await self._check_style(code)
        functional_issues = await self._check_functionality(code)
        optimization_suggestions = await self._check_optimization(code)
        
        # 合并所有问题
        all_issues = syntax_issues + style_issues + functional_issues
        result["issues"] = all_issues
        result["suggestions"] = optimization_suggestions
        
        # 计算分数
        for issue in all_issues:
            severity = issue["severity"]
            result["severity_distribution"][severity] += 1
            
            # 扣分
            if severity == "HIGH":
                result["score"] -= 10
            elif severity == "MEDIUM":
                result["score"] -= 5
            else:
                result["score"] -= 2
        
        result["score"] = max(0, result["score"])
        
        # 决定状态
        if result["severity_distribution"]["HIGH"] > 0:
            result["status"] = "rejected"
        elif result["score"] < 70:
            result["status"] = "needs_improvement"
        else:
            result["status"] = "approved"
        
        return result
    
    async def _check_syntax(self, code: str) -> List[Dict[str, Any]]:
        """检查语法问题"""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            # 检查敏感列表
            if "always @" in line and "*" not in line:
                # 简化检查：如果不是 always @*，检查是否完整
                if "posedge" not in line and "negedge" not in line:
                    issues.append({
                        "type": "incomplete_sensitivity_list",
                        "severity": "HIGH",
                        "line": i + 1,
                        "description": "Potentially incomplete sensitivity list",
                        "suggestion": "Use 'always @*' or list all read signals"
                    })
        
        return issues
    
    async def _check_style(self, code: str) -> List[Dict[str, Any]]:
        """检查代码风格"""
        issues = []
        
        # 检查命名约定
        if "clk" in code and "clock" in code:
            issues.append({
                "type": "inconsistent_naming",
                "severity": "LOW",
                "description": "Inconsistent clock signal naming",
                "suggestion": "Use consistent naming: either 'clk' or 'clock'"
            })
        
        return issues
    
    async def _check_functionality(self, code: str) -> List[Dict[str, Any]]:
        """检查功能问题"""
        issues = []
        
        # 检查复位逻辑
        if "reset" in code.lower() or "rst" in code.lower():
            if "always @" in code and "reset" in code:
                # 简单检查是否处理了复位
                if code.count("if") > 0 and "reset" not in code[code.find("if"):code.find("if")+50]:
                    issues.append({
                        "type": "missing_reset_handling",
                        "severity": "MEDIUM",
                        "description": "Reset signal may not be properly handled",
                        "suggestion": "Ensure reset is checked in sequential logic"
                    })
        
        return issues
    
    async def _check_optimization(self, code: str) -> List[str]:
        """检查优化机会"""
        suggestions = []
        
        # 基于智能体的分析能力提供建议
        if self.agent.profile.capabilities.get(AgentCapability.CODE_OPTIMIZATION, 0) > 0.7:
            # 高级优化建议
            if code.count("always") > 3:
                suggestions.append("Consider combining related always blocks for better readability")
            
            if "case" in code and "parallel_case" not in code:
                suggestions.append("Consider using synthesis pragmas for case statements")
        
        return suggestions