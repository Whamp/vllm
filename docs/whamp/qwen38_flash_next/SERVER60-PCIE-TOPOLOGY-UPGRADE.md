# server60 PCIe topology and motherboard upgrade research

Date: 2026-08-29

Status: research only. No BIOS, hardware, service, or benchmark configuration was changed.

## Decision

Do not buy a replacement motherboard yet.

server60's ASUS ROG Zenith Extreme and Threadripper 2950X cannot give all four GPUs x16 links. The documented four-GPU layout is PCIe 3.0 x16/x8/x16/x8. That is a platform limit worth replacing if matched measurements show NCCL bandwidth matters.

The current machine is worse than that documented limit: x4/x16/x8/x16. The x4 GPU most likely occupies the board's `PCIEX8/X4_4` slot while that slot is configured for x4 mode. ASUS documents that x4 mode leaves the U.2 port enabled, while x8 mode disables U.2. server60's only NVMe device is a Samsung 970 PRO M.2 drive, not a native U.2 drive. After confirming its physical connection, the first experiment should switch `PCIEX8/X4_4 Bandwidth` to `X8 mode`, verify the link under load, and rerun matched NCCL and vLLM measurements.

A replacement X399 motherboard would retain the same CPU lane budget and PCIe 3.0 ceiling. The cheapest path is therefore to keep this platform and recover the missing x4 lanes first.

If measurement later justifies a platform change, RAM reuse changes the ranking. server60 currently has four 16 GB Kingston `KHX2400C15/16G` non-ECC DDR4 UDIMMs, and Will has an identical second four-module kit not yet installed. Standard EPYC 7001/7002/7003 SP3 platforms require registered DDR4 and would strand all eight modules. Credible DDR4-UDIMM candidates include a used WRX80 board with a low-core Threadripper Pro, TRX40 with a 3960X, or an Intel ASUS WS X299 SAGE with a low-cost Core i9-10900X. The X299 board uses two PLX PCIe switches to present four PCIe 3.0 x16 GPU endpoints from 48 CPU lanes, so it needs application-level P2P and NCCL validation before purchase.

## Live server60 evidence

The following values were collected from server60 on 2026-08-29.

```text
Motherboard: ASUS ROG ZENITH EXTREME Rev 1.xx
CPU:         AMD Ryzen Threadripper 2950X
GPUs:        4 x NVIDIA GeForce RTX 3090
Memory:      4 x 16 GB Kingston KHX2400C15/16G DDR4-2400 UDIMM installed
Purchased:   4 x 16 GB identical KHX2400C15/16G modules, not yet installed
```

`nvidia-smi topo -m` reports no NVLink. Every GPU pair crosses either a PCIe host bridge (`PHB`) or host bridges within the same NUMA node (`NODE`). CUDA peer reads and writes report `OK` for every directed GPU pair.

The negotiated widths are asymmetric:

| Linux GPU | PCI bus ID | Root port | Negotiated width | Link generation under load |
| --- | --- | --- | ---: | ---: |
| GPU 0 | `0000:08:00.0` | `0000:00:01.3` | x4 | PCIe 3.0 |
| GPU 1 | `0000:09:00.0` | `0000:00:03.1` | x16 | PCIe 3.0 |
| GPU 2 | `0000:41:00.0` | `0000:40:01.3` | x8 | PCIe 3.0 |
| GPU 3 | `0000:42:00.0` | `0000:40:03.1` | x16 | PCIe 3.0 |

At idle the links reduce speed to PCIe Gen 1, then return to Gen 3 under load. Width does not change with that power-management transition. `lspci -vv` marks GPU 0's x4 and GPU 2's x8 widths as downgraded relative to each RTX 3090 endpoint's x16 capability.

The theoretical one-direction payload rates, before protocol overhead, are approximately:

| Link | Theoretical rate |
| --- | ---: |
| PCIe 3.0 x4 | 3.94 GB/s |
| PCIe 3.0 x8 | 7.88 GB/s |
| PCIe 3.0 x16 | 15.75 GB/s |
| PCIe 4.0 x16 | 31.51 GB/s |

These are link limits, not measured NCCL bandwidth. An all-reduce can instead be limited by fixed collective latency, synchronization, rank imbalance, topology, or exposed communication. The Qwen decode trace must supply collective counts and payload sizes before the x4 link can be called the application bottleneck.

### Reproduction commands

```bash
ssh server60 'sudo dmidecode -t baseboard'
ssh server60 'lscpu'
ssh server60 'nvidia-smi topo -m'
ssh server60 'nvidia-smi topo -p2p r; nvidia-smi topo -p2p w'
ssh server60 'nvidia-smi --query-gpu=index,pci.bus_id,name,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader'
ssh server60 'lspci -tv'
ssh server60 'sudo lspci -s 08:00.0 -vv'
ssh server60 'sudo lspci -s 09:00.0 -vv'
ssh server60 'sudo lspci -s 41:00.0 -vv'
ssh server60 'sudo lspci -s 42:00.0 -vv'
ssh server60 'lsblk -o NAME,MODEL,SERIAL,SIZE,TRAN,MOUNTPOINTS'
```

## What the current board should provide

The ASUS ROG Zenith Extreme manual, section 1.1.5, specifies these GPU layouts:

| GPU count | Documented lane layout |
| ---: | --- |
| 1 | x16 |
| 2 | x16/x16 |
| 3 | x16/x8/x16 |
| 4 | x16/x8/x16/x8 |

The same manual says:

> The PCIE_X8/X4_4 slot shares bandwidth with U.2. In 4-Way configuration, if the PCIE_X8/X4_4 is used in x8 mode, U.2 port will be disabled.

Its BIOS documentation provides these choices:

- `X8 mode`: `PCIEX8/X4_4` runs at x8 and U.2 is disabled.
- `X4 mode`: `PCIEX8/X4_4` runs at x4 and U.2 is enabled.

server60's x16/x8/x16/x4 pattern matches the documented four-GPU layout with the last slot in x4 mode. This is strong evidence, but the firmware setting and physical slot assignment still need direct inspection before changing anything. A riser or contact problem can also train an x8 slot at x4.

The installed Samsung 970 PRO 1 TB reports as PCIe 3.0 x4 and contains the root filesystem and `/mnt/models`. Samsung specifies that drive as M.2 2280. It is therefore unlikely to require the motherboard's U.2 connector unless an M.2-to-U.2 adapter is installed. Confirm the physical connection before disabling U.2.

AMD documents 64 CPU PCIe 3.0 lanes for the 2950X, with four reserved for the X399 chipset and at most 60 exposed to devices. The ASUS board spends its lane budget on two x16 GPU slots, two x8 GPU slots, CPU-attached NVMe, and the chipset connection. Another X399 board cannot add PCIe 4.0 support to this CPU and is unlikely to justify the migration.

## Expected value of the free x4-to-x8 correction

If a large all-reduce is set by GPU 0's x4 link, moving that slot to x8 doubles that link's theoretical rate. It does not double end-to-end decode throughput.

### Historical PYNCCL ceiling, now superseded

The original FP8-QSA service used PYNCCL and measured 43.77 token/s. Its matched trace put BF16 ring all-reduce at 62.5% of summed c=1 kernel time and about 4.221 ms of summed collective kernel time per generated token. This evidence led to a software topology fix before a hardware purchase.

The parent Qwen session promoted island-aware hierarchical all-reduce for rank islands `0,1;2,3` on the existing X399 host. With the checkpoint, FP8 QSA cache, PLE, context, batch budget, and power policy unchanged, it measured:

| Measurement | PYNCCL | Hierarchical | Change |
| --- | ---: | ---: | ---: |
| Single-stream decode | 43.77 token/s | 50.34 token/s | +15.0% |
| Cache-busted prefill | 1,529.25 token/s | 1,538.14 token/s | +0.6% |
| Concurrency-2 aggregate | 53.25 token/s | 59.00 token/s | +10.8% |

The exact-size four-GPU mechanism gate measured 2,560-element BF16 all-reduce at 80.90 microseconds for hierarchical versus 107.52 microseconds for NCCL. This small payload is mainly a latency and synchronization problem, not a bulk-link-rate problem. The prior calculation that projected approximately 51 token/s from an 8× collective speedup is therefore obsolete: software already reached 50.34 token/s without changing PCIe generation.

The incremental PCIe 4.0 benefit must now be bounded from a fresh trace of the hierarchical production path. Until that trace separates remaining transfer time from fixed synchronization and kernel work, a large additional decode gain from WRX80 is unsupported. PCIe 4.0 remains attractive for uniform topology, larger prefill or batch collectives, eight-channel PLE bandwidth, remote management, and future expansion, but not as a demonstrated path from 50 to 66–70 token/s.

## Upgrade choices under the RAM-reuse constraint

### Lowest cost: keep X399 and recover x8

This is the current recommendation. It reuses the 2950X, all DDR4, storage, cooler, and board. If the fourth slot reaches x8, the weakest link rises from PCIe 3.0 x4 to PCIe 3.0 x8 without buying anything.

The additional 64 GB kit has the same `KHX2400C15/16G` part number as the four installed dual-rank modules. Populate the board according to ASUS's eight-DIMM diagram, confirm all 128 GB trains at DDR4-2400, and run an extended memory test before returning the inference service to production. Install the RAM and change PCIe lane mode in separate maintenance windows so either failure has one cause.

### Preferred PCIe 4.0 and expansion option: WRX80 with a low-core CPU

WRX80 is the only platform class found that combines PCIe 4.0, more than four full-length GPU slots, and official non-ECC DDR4 UDIMM support. Three boards fit server60's constraints:

| Board | Full-length PCIe 4.0 slots | Electrical widths | Non-ECC UDIMM | Remote management |
| --- | ---: | --- | --- | --- |
| ASUS Pro WS WRX80E-SAGE SE WIFI | 7 | 7 × x16 | Explicitly supported | ASMB9-iKVM |
| ASRock WRX80 Creator | 7 | 5 × x16 + 2 × x8 | Explicitly supported | Dedicated IPMI |
| Supermicro M12SWA-TF | 6 | 6 × x16 | Explicitly supported, up to 256 GB | BMC/IPMI |

The ASUS board is the cleanest maximum-expansion choice. It provides seven CPU-connected PCIe 4.0 x16 slots without the dual-PLX compromise of X299. Its BMC also makes future BIOS and console recovery remotely manageable. Supermicro is the lower-slot alternative worth watching on the used market: one documented used listing was about $600, versus roughly $800–1,250 examples for ASUS or ASRock boards. These prices are volatile seller examples, not a purchasing quote.

The CPU need not be large. A used 12-core Threadripper Pro 3945WX or 16-core 3955WX exposes the same 128-lane PCIe 4.0 and eight-channel DDR4 fabric as the expensive high-core-count models. Eight installed DIMMs populate all eight memory channels, doubling theoretical host-memory bandwidth over server60's current four-channel platform at the same DDR4-2400 speed. That may also help the CPU-resident Qwen PLE path, although this is not yet a measured serving gain.

Used Threadripper Pro CPUs require special care: Lenovo and some Dell systems permanently vendor-lock CPUs with AMD Platform Secure Boot. Do not buy a CPU described as Lenovo-locked, P620-pulled, or vendor-locked. Require a seller to attest that it is unlocked and tested in a retail ASUS, ASRock, or Supermicro WRX80 board, with a return policy.

This changes the weakest current endpoint from PCIe 3.0 x4 to PCIe 4.0 x16, an eightfold theoretical per-link increase, and makes four GPU links uniform. All eight Kingston modules are the same `KHX2400C15/16G` part, but successful training is not guaranteed until verified against the chosen board and tested at DDR4-2400.

Seven slots do not mean seven air-cooled RTX 3090s fit directly on the board. Most 3090s occupy 2.5 to 3 slots. Five to seven cards require a proper open-frame or rack chassis, Gen4-qualified risers or cables, mechanically supported cards, auxiliary slot power, sufficient PSU capacity, and a suitable mains circuit. At the fixed 230 W safety limit, six GPUs alone draw up to 1.38 kW before CPU, memory, drives, fans, and conversion losses. Software topology matters too: TP/EP sizes of 5, 6, or 7 may not divide Qwen's head and expert counts cleanly. Treat the extra slots as expansion capacity, not an immediate promise that every GPU count is useful.

### Cheaper-board DDR4 option: TRX40

TRX40 still requires another Threadripper, but it accepts ordinary non-ECC DDR4 UDIMMs and may be cheaper than WRX80 on the used market. The platform's entry CPU is the 24-core 3960X, so some of the purchase pays for CPU cores this host does not need.

The ASRock TRX40 Creator, Gigabyte TRX40 AORUS XTREME, ASUS ROG Zenith II Extreme, and MSI Creator TRX40 document four-GPU PCIe 4.0 x16/x8/x16/x8 layouts and non-ECC DDR4 UDIMM support. Prefer a board whose M.2 placement does not reduce the fourth GPU slot. ASUS documents that an active `M.2_2` can reduce its fourth slot to x4, repeating the current problem.

PCIe 4.0 x8 has the same theoretical payload rate as PCIe 3.0 x16. A correct TRX40 layout therefore doubles the weakest-link rate relative to the corrected X399 x8 layout, while WRX80 x16 doubles it again.

### PCIe 3.0 fallback only: Intel X299 with PLX switches

The ASUS WS X299 SAGE officially supports eight non-ECC unbuffered DDR4 DIMMs, 256 GB with supported CPUs, and four GPUs at x16 endpoint width. A used Core i9-10900X supplies 10 cores, 48 PCIe 3.0 lanes, four memory channels, and official support for 256 GB. ASUS validates it on this board with BIOS 2002 or newer.

The board creates four x16 endpoints with two PLX PEX8747 switches. It does not create PCIe bandwidth at the CPU. Same-switch peer traffic can remain inside a PLX switch, while traffic crossing switches shares their upstream CPU links. ACS firmware settings can also redirect peer traffic through the root complex. An NVIDIA forum report on this board improved GPUDirect throughput from about 38 to 92 Gbit/s by disabling ACS on the PLX ports, which proves the switch-local path can work but also shows the tuning risk. That report used a Tesla T4 and BlueField-2, not four RTX 3090s or NCCL.

This is the strongest inexpensive non-Threadripper lead because it reuses all 128 GB and accepts a comparatively cheap CPU, but it does not meet the preferred PCIe 4.0 requirement. It remains a PCIe 3.0 switch topology. Buy only from a returnable source and only after finding a bundle price low enough to justify a direct `p2pBandwidthLatencyTest`, `nccl-tests`, and vLLM trial. The older ASUS X99-E WS offers the same general trick with cheaper Xeon E5 or Core i7 CPUs, but field reports include PLX-related CUDA instability. It is too old and uncertain for the production recommendation.

### EPYC UDIMM hack research: no credible SP3 success found

Official ASRock Rack ROMED8-2T, Supermicro H11SSL, and Gigabyte MZ32-AR0 specifications list RDIMM, LRDIMM, or NVDIMM only. Their manuals and QVLs do not list UDIMMs.

Community reports match the manuals. A Level1Techs H11SSL owner diagnosed a no-POST system as UDIMMs and booted only after replacing them with RDIMMs. Separate Level1Techs and ServeTheHome discussions report no known SP3 board that trains unbuffered memory. Searches for Chinese SP3 workstation boards found the same RDIMM/LRDIMM requirement even when storefront titles only said `DDR4`.

This is not an ECC toggle. `KHX2400C15/16G` is an unbuffered module, while SP3 expects a register clock driver on each RDIMM. A modified BIOS cannot safely turn one electrical interface into the other. EPYC 4004/4005 uses UDIMMs on AM5, but that platform lacks the PCIe lanes for four high-bandwidth GPUs and uses DDR5 rather than this DDR4 kit.

WRX90 supplies up to 128 PCIe 5.0 lanes but requires a new Threadripper Pro 7000 WX CPU and DDR5 registered memory. The RTX 3090 stops at PCIe 4.0, so WRX90 gives these cards no higher link rate than WRX80. It is the wrong cost profile for this inference-only host.

## Measurement plan before purchase

1. Finish the current compilation A/B and restore production before touching firmware.
2. Record the existing firmware value for `PCIEX8/X4_4 Bandwidth` and photograph or otherwise identify the SSD's physical connector.
3. If the SSD is not using U.2, select `X8 mode`, boot, and confirm all expected storage and GPUs appear.
4. Under GPU load, verify widths of x16/x8/x16/x8. Check kernel logs and PCIe AER counters for link errors.
5. Run NVIDIA `p2pBandwidthLatencyTest` for every directed GPU pair across small and large payloads.
6. Run `nccl-tests all_reduce_perf` at the exact collective sizes observed in the Qwen decode trace, then run its normal size sweep.
7. Repeat the unprofiled cached c=1 and c=2 Qwen benchmarks with the service, image, model, clocks, power limits, and request inputs unchanged.
8. Compare collective time, achieved `algbw` and `busbw`, inter-token latency, and end-to-end throughput. Do not buy a platform based only on negotiated width.

Proceed to platform pricing only if the x8 correction leaves NCCL bandwidth on the critical path and the estimated application gain is worth replacing the board and CPU. All eight RAM modules carry the same part number, but a prospective board still needs to support eight dual-rank 16 GB non-ECC UDIMMs at DDR4-2400.

## Primary sources

- ASUS, [ROG Zenith Extreme user manual, revised edition V2](https://dlcdnets.asus.com/pub/ASUS/mb/socketTR4/ROG_ZENITH_EXTREME/E13369_ROG_ZENITH_EXTREME_UM_V2_WEB.pdf), expansion slots section 1.1.5 and BIOS `PCIEX8/X4_4 Bandwidth` setting.
- AMD, [Socket TR4 X399 motherboards](https://www.amd.com/en/products/processors/chipsets/str4.html), CPU and chipset lane accounting.
- AMD, [2nd Generation Ryzen Threadripper launch specifications](https://www.amd.com/en/newsroom/press-releases/2018-8-6--world-record-breaking-2nd-generation-amd-ryzen-t.html), 2950X PCIe generation and lane count.
- Samsung, [970 PRO data sheet](https://semiconductor.samsung.com/resources/data-sheet/Samsung_NVMe_SSD_970_PRO_Data_Sheet_Rev.1.0.pdf), M.2 2280 and PCIe 3.0 x4 interface.
- NVIDIA, [GA102 architecture whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf), RTX 3090 PCIe 4.0 board interface.
- NVIDIA, [`p2pBandwidthLatencyTest`](https://github.com/NVIDIA/cuda-samples/tree/v12.9/Samples/5_Domain_Specific/p2pBandwidthLatencyTest), directed P2P validation and measurement.
- NVIDIA, [`nccl-tests`](https://github.com/NVIDIA/nccl-tests), collective correctness and performance measurement.
- ASUS, [Pro WS WRX80E-SAGE SE WIFI specifications](https://www.asus.com/motherboards-components/motherboards/workstation/pro-ws-wrx80e-sage-se-wifi/techspec/) and [user manual](https://dlcdnets.asus.com/pub/ASUS/mb/SocketTRX4/Pro_WS_WRX80E-SAGE_SE_WIFI/E19401_Pro_WS_WRX80E-SAGE_SE_WIFI_UM_V2_WEB.pdf).
- AMD, [Threadripper Pro 5000 WX launch specifications](https://www.amd.com/en/newsroom/press-releases/2022-3-8-new-amd-ryzen-threadripper-pro-5000-wx-series-proc.html), 128 PCIe 4.0 lanes.
- ASRock, [TRX40 Creator specifications](https://www.asrock.com/mb/AMD/TRX40%20Creator/index.asp), PCIe 4.0 x16/x8/x16/x8 and non-ECC DDR4 UDIMM support.
- Gigabyte, [TRX40 AORUS XTREME specifications](https://www.gigabyte.com/us/Motherboard/TRX40-AORUS-XTREME-rev-11/sp), PCIe 4.0 x16/x8/x16/x8 and non-ECC DDR4 UDIMM support.
- ASUS, [ROG Zenith II Extreme specifications](https://rog.asus.com/motherboards/rog-zenith/rog-zenith-ii-extreme-model/spec/), PCIe 4.0 x16/x8/x16/x8, non-ECC DDR4 UDIMM support, and fourth-slot M.2 sharing.
- MSI, [Creator TRX40 specifications](https://www.msi.com/Motherboard/creator-trx40/Specification), PCIe 4.0 x16/x8/x16/x8 and unbuffered DDR4 support.
- ASUS, [Pro WS WRX80E-SAGE SE WIFI specifications](https://www.asus.com/motherboards-components/motherboards/workstation/pro-ws-wrx80e-sage-se-wifi/techspec/) and [user manual](https://dlcdnets.asus.com/pub/ASUS/mb/SocketTRX4/Pro_WS_WRX80E-SAGE_SE_WIFI/E19401_Pro_WS_WRX80E-SAGE_SE_WIFI_UM_V2_WEB.pdf), seven CPU-connected PCIe 4.0 x16 slots and non-ECC DDR4 UDIMM support.
- ASRock, [WRX80 Creator specifications](https://www.asrock.com/MB/AMD/WRX80%20Creator/), five PCIe 4.0 x16 plus two x8 links and non-ECC UDIMM support.
- Supermicro, [M12SWA-TF specifications](https://www.supermicro.com/en/products/motherboard/M12SWA-TF), six PCIe 4.0 x16 slots and up to 256 GB non-ECC UDIMM.
- AMD, [Threadripper Pro 3945WX launch specifications](https://www.amd.com/en/newsroom/press-releases/2020-7-14-amd-announce-world-s-first-64-core-pro-workstation.html), 128 PCIe 4.0 lanes and eight DDR4 channels.
- ServeTheHome, [Lenovo Threadripper Pro AMD PSB locking report](https://www.servethehome.com/lenovo-is-using-amd-psb-to-vendor-lock-amd-cpus/), permanent used-CPU compatibility risk.
- ASUS, [WS X299 SAGE specifications](https://www.asus.com/motherboards-components/motherboards/workstation/ws-x299-sage/techspec/) and [user manual](https://dlcdnets.asus.com/pub/ASUS/mb/Socket2066/WS_X299_SAGE/Manual/E16044_WS_X299_SAGE_UM_V4_WEB.pdf), non-ECC DDR4 UDIMMs, dual PLX switches, and quad-x16 endpoints.
- Intel, [Core i9-10900X specifications](https://www.intel.com/content/www/us/en/products/compare.html?productIds=%2F198017%2C198019%2C198012%2C198014), 48 PCIe 3.0 lanes and 256 GB non-ECC memory support.
- NVIDIA Developer Forums, [ASUS WS X299 SAGE PLX ACS case report](https://forums.developer.nvidia.com/t/gpudirect-rdma-bandwidth-bottleneck-38gbps-on-asus-ws-x299-sage-10g-with-tesla-t4-bluefield-2/355218), measured switch-local GPUDirect recovery after ACS changes.
- ASRock Rack, [ROMED8-2T specifications](https://www.asrockrack.com/general/productdetail.asp?Model=ROMED8-2T), seven PCIe 4.0 x16 slots and registered-memory-only support.
- Supermicro, [H11SSL-i specifications](https://www.supermicro.com/en/products/motherboard/H11SSL-i), registered-memory-only support.
- Level1Techs, [EPYC no-POST case caused by UDIMMs](https://forum.level1techs.com/t/solved-epyc-issues-no-post/194509), successful boot after replacing UDIMMs with RDIMMs.
- AMD, [Threadripper Pro 7000 WX and WRX90 launch specifications](https://www.amd.com/en/newsroom/press-releases/2023-10-19-amd-introduces-new-amd-ryzen-threadripper-7000-ser.html), up to 128 PCIe 5.0 lanes.
- ASUS, [Pro WS WRX90E-SAGE SE specifications](https://www.asus.com/motherboards-components/motherboards/workstation/pro-ws-wrx90e-sage-se/techspec/) and [manual](https://dlcdnets.asus.com/pub/ASUS/mb/SocketsTR5/Pro_WS_WRX90E-SAGE_SE/E22564_Pro_WS_WRX90E-SAGE_SE_EM_WEB.pdf).
