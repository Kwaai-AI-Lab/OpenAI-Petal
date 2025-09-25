# KwaaiNet NodeManager

Dynamic process orchestrator for KwaaiNet distributed inference nodes. Manages multiple Petals server processes to enable runtime model loading and efficient resource utilization.

## Overview

The NodeManager transforms KwaaiNet from a static single-model node into a dynamic multi-model node that can:
- Start with minimal resources (~100MB vs ~2GB traditional installation)
- Load models on-demand when network requests arrive
- Manage multiple Petals processes for different models simultaneously
- Provide intelligent resource scheduling and failure recovery

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Network       │    │   NodeManager    │    │   Petals        │
│   Requests      │───▶│   Orchestrator   │───▶│   Processes     │
│                 │    │                  │    │                 │
│ • Model A       │    │ • Process Mgmt   │    │ • Process 1: A  │
│ • Model B       │    │ • Health Monitor │    │ • Process 2: B  │
│ • Blocks X-Y    │    │ • Resource Mgmt  │    │ • Process N...  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Project Status

**Phase**: Requirements and Architecture Design
**Version**: 0.1.0-alpha
**Target Platforms**: macOS, Linux, Windows (planned)

## Quick Start

*Implementation in progress - requirements phase*

```bash
# Future usage (not yet implemented)
cd NodeManager
pip install -e .
kwaainet-node start --minimal
# Node starts with networking only, loads models on demand
```