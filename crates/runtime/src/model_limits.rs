//! Re-export shim: the model-limits table moved to [`nca_common::model_limits`]
//! so the provider layer in `nca-core` (which cannot depend on `nca-runtime`)
//! can apply capability-aware `max_tokens` clamping. All existing
//! `nca_runtime::model_limits` paths keep working unchanged.

pub use nca_common::model_limits::*;
