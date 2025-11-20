# OpenAI-Petal: Legacy Status & Maintenance Mode

**Document Date:** 2025-11-20
**Project Version:** v0.6.3 (Final Feature Release)
**Status:** MAINTENANCE MODE
**End of Life:** Q2 2027 (Estimated)

---

## 🎯 Mission Accomplished

OpenAI-Petal has successfully served its purpose as a **technical sandbox** for distributed AI inference. The project proved key concepts and established working infrastructure for cross-platform node deployment.

**Key Achievements:**
- ✅ Production-ready installers (Linux, macOS)
- ✅ Robust daemon management with health monitoring
- ✅ Auto-calibration and auto-update systems
- ✅ ~100 active nodes in developer/enthusiast community
- ✅ 104 Python files, comprehensive implementation

Thank you to all contributors and users who helped validate this technology!

---

## 📢 MAINTENANCE-ONLY MODE (Effective Immediately)

As of **November 20, 2025**, OpenAI-Petal enters maintenance-only mode:

### What This Means

**✅ WILL CONTINUE:**
- Critical bug fixes (crashes, data loss, security vulnerabilities)
- Community support via GitHub Issues
- Existing nodes will continue functioning
- Documentation maintenance

**❌ WILL NOT HAPPEN:**
- New features
- Dependency updates (too risky given CVE constraints)
- Performance optimizations
- Platform expansions (Windows remains unsupported)
- Architecture changes

### Version Status

- **v0.6.3** - Final feature release ✅
- **v0.6.4+** - Critical fixes only (if needed)
- **v1.0** - Will NOT be released

---

## 🚨 Why Maintenance Mode?

### Strategic Pivot to KwaaiNet

The Kwaai project is pivoting from **developer tooling** to **mass consumer adoption**. This requires fundamental architectural changes that cannot be achieved within OpenAI-Petal's Python/Petals framework.

### Technical Constraints

OpenAI-Petal faces architectural limitations that block mass adoption:

1. **Security:** 8 CVEs in transformers dependency (cannot fix without breaking Petals)
2. **Onboarding time:** 30-45 minutes (model download bottleneck)
3. **Browser support:** Architecturally impossible (Python runtime requirement)
4. **Dependency hell:** Every update breaks (transformers, triton, bitsandbytes conflicts)
5. **Scale ceiling:** ~10K technical users maximum (Python/Docker barrier)

These are not bugs to fix - they are fundamental constraints of the chosen technology stack.

### Official Strategy Statement

From CEO Reza Rassool's MASS_ADOPTION_STRATEGY.md (September 2025):

> **"FREEZE CURRENT TOOLING - No more Python/Docker development. Sufficient for developer ecosystem."**

> **"This strategy pivots KwaaiNet from complex developer tooling to consumer-simple mass adoption."**

---

## 🌐 The Future: KwaaiNet

Development efforts are now focused on **KwaaiNet** - a ground-up rewrite for 1B+ users.

### KwaaiNet Vision

**Architecture:**
- **Language:** Rust (compiles to WASM for browsers)
- **ML Framework:** Candle (WebGPU support)
- **Distribution:** Browser extensions, mobile apps, single-binary desktop
- **Onboarding:** <10 seconds (vs 30-45 minutes in OpenAI-Petal)
- **Security:** Clean slate, no Python dependency trap

**Target Platforms:**
- ✅ Browser (Chrome, Firefox, Safari) - Phase 1
- ✅ Mobile (iOS, Android) - Phase 2
- ✅ Desktop (Windows, macOS, Linux) - Phase 3
- ✅ Embedded (Routers, IoT devices) - Phase 4

**Business Model:**
- Triple service: AI Compute + Private Storage (Verida) + Data Sovereignty
- VDA token economics
- Environmental incentives (carbon tracking, renewable energy bonuses)

### Timeline

| Milestone | Date | Status |
|-----------|------|--------|
| Hackathon launch (6 challenges, 3M VDA prizes) | Q1 2026 | Planned |
| Browser extension | Q1 2026 | In development |
| Mobile apps (iOS/Android) | Q2-Q3 2026 | Planned |
| 1M+ nodes target | Q3 2026 | Goal |
| OpenAI-Petal deprecation notice | Q4 2026 | Scheduled |
| OpenAI-Petal final shutdown | Q2 2027 | Estimated |

**GitHub:** https://github.com/Kwaai-AI-Lab/KwaaiNet

---

## 📅 Deprecation Timeline

### Phase 1: Maintenance Mode (NOW - Q3 2026)

**Status:** OpenAI-Petal remains functional

**Actions:**
- Continue operating existing nodes
- Community support for troubleshooting
- Critical bug fixes only
- No new development

**User Impact:** None - everything continues working

### Phase 2: Deprecation Notice (Q4 2026)

**Status:** 6-month wind-down announced

**Actions:**
- Official deprecation announcement
- Migration tooling released
- Bonus rewards for early migrators
- Documentation archived

**User Impact:** Start planning migration to KwaaiNet

### Phase 3: Final Shutdown (Q2 2027)

**Status:** OpenAI-Petal services discontinued

**Actions:**
- Bootstrap servers decommissioned
- Map API stops displaying OpenAI-Petal nodes
- GitHub repository archived (read-only)
- Final backup of all documentation

**User Impact:** Must migrate to KwaaiNet or other platforms

---

## 🔄 Migration Path

### For Current OpenAI-Petal Users

**Option 1: Early Migration to KwaaiNet (Recommended)**

**When:** Q1-Q2 2026 (when KwaaiNet browser extension launches)

**Benefits:**
- Bonus rewards for early adopters
- Free KwaaiNet Pro tier (6 months)
- Transfer contribution history
- Help shape KwaaiNet development

**Process:**
1. Install KwaaiNet browser extension (Chrome/Firefox)
2. Link accounts using migration tool
3. Verify new node on map.kwaai.ai
4. Optionally keep OpenAI-Petal running until Q2 2027

**Option 2: Download KwaaiNet Native App**

**When:** Q2-Q3 2026 (when mobile/desktop apps launch)

**Benefits:**
- Higher performance than browser mode
- Full GPU acceleration
- 24/7 operation

**Process:**
1. Download from App Store / Google Play / kwaainet.io
2. One-click setup (vs 8-minute install in OpenAI-Petal)
3. Automatic config migration

**Option 3: Continue with OpenAI-Petal Until Shutdown**

**When:** Through Q2 2027

**Process:**
- No action required until deprecation notice (Q4 2026)
- Maintain current setup
- Monitor for migration announcements

---

## ❓ FAQ

### Q: Will my node stop working immediately?

**A:** No. OpenAI-Petal nodes will continue functioning normally through at least Q2 2027.

### Q: Can I still install OpenAI-Petal on new machines?

**A:** Yes, installers remain available. However, we recommend waiting for KwaaiNet browser extension (Q1 2026) for better experience.

### Q: What happens to my contribution history?

**A:** Migration tools will transfer your stats, uptime, and reputation to KwaaiNet. Early migrators get bonus rewards.

### Q: Will you fix the Windows installer?

**A:** No. Windows support will not be restored in OpenAI-Petal. Wait for KwaaiNet cross-platform apps (Q2-Q3 2026).

### Q: Can I continue using OpenAI-Petal after 2027?

**A:** Technically yes (it's open source), but bootstrap servers will be offline, making network participation impossible. You could fork and run private swarms.

### Q: Why can't you just update OpenAI-Petal instead of rewriting?

**A:** OpenAI-Petal's Python/Petals foundation cannot support:
- Browser deployment (Python → WASM impossible)
- Instant onboarding (model download bottleneck)
- Security fixes (CVEs locked by Petals dependency)
- Mass scale (10K user ceiling)

These require architecture-level changes, not updates.

### Q: Will KwaaiNet be compatible with OpenAI-Petal nodes?

**A:** No. KwaaiNet uses a different P2P protocol and model format. During transition, both networks operate independently.

### Q: What if I want to keep running Python-based nodes?

**A:** You can fork OpenAI-Petal and maintain your own version. Kwaai will not provide official support, but the codebase is open source (CC-BY-4.0).

---

## 🤝 Thank You

To all developers, contributors, and early adopters who tested OpenAI-Petal:

**You helped prove distributed AI inference is viable.**

Your feedback shaped our understanding of:
- Real-world deployment challenges
- User onboarding friction points
- Hardware compatibility needs
- Network reliability requirements

This knowledge directly informed KwaaiNet's design.

**The future is bright. Join us in building KwaaiNet for the next 1 billion users.**

---

## 📞 Support

### For OpenAI-Petal Issues

- **GitHub Issues:** https://github.com/Kwaai-AI-Lab/OpenAI-Petal/issues
- **Community:** Kwaai Slack (kwaaiailab.slack.com)
- **Response Time:** Best effort (maintenance mode)

### For KwaaiNet Information

- **GitHub:** https://github.com/Kwaai-AI-Lab/KwaaiNet
- **Website:** https://kwaai.ai
- **Twitter:** @KwaaiAI
- **Discord:** discord.gg/kwaai

---

## 📚 Documentation Archive

All OpenAI-Petal documentation will be preserved:

- **Installation guides** - Reference for historical context
- **Architecture docs** - Learning resource for distributed systems
- **Troubleshooting** - Community knowledge base
- **Version history** - Development chronicle (v0.1.0 → v0.6.3)

**Archive Location:** https://github.com/Kwaai-AI-Lab/OpenAI-Petal/wiki

---

## 🔒 Security Notice

### Known Vulnerabilities (Will NOT Be Fixed)

OpenAI-Petal depends on `transformers==4.43.1` which has:

- **CVE-2025-1194** (ReDoS in tokenizers) - 🔴 CRITICAL
- **CVE-2025-2099** (ReDoS in testing_utils) - 🔴 CRITICAL
- **CVE-2024-11392** (Code injection) - 🟠 HIGH
- **CVE-2024-11393** (Deserialization) - 🟠 HIGH
- **CVE-2024-11394** (Path traversal) - 🟠 HIGH

**Why not fix?** Updating transformers breaks Petals compatibility entirely.

**Mitigation:**
- Run in isolated environments (Docker, VMs)
- Don't process untrusted input through tokenizers
- Use firewalls to limit exposure
- Monitor for unusual CPU usage (ReDoS indicators)

**Long-term solution:** Migrate to KwaaiNet (clean security slate)

---

## 📜 License

OpenAI-Petal remains open source under **CC-BY-4.0 license**.

You are free to:
- ✅ Use for personal/commercial purposes
- ✅ Fork and modify
- ✅ Redistribute

Requirements:
- ✅ Give appropriate credit
- ✅ Indicate changes made

**No warranty or support provided in maintenance mode.**

---

## 🎯 Closing Thoughts

OpenAI-Petal was never meant to be the final product - it was a **stepping stone** to understand distributed AI inference.

**Mission accomplished:**
- Proved technical feasibility ✅
- Identified adoption barriers ✅
- Informed KwaaiNet design ✅
- Built engaged developer community ✅

Now we build for 1 billion users.

**See you in KwaaiNet.** 🚀

---

**Document Version:** 1.0
**Last Updated:** 2025-11-20
**Maintained By:** Kwaai Core Team
