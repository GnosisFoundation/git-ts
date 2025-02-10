use candle::safetensors::{load, save};
use candle::{Device, Tensor};
use chrono::prelude::*;
use safetensors::SafeTensors;
use sha2::{Digest, Sha512};
use time::UnixTimestamp;

use std::collections::HashMap;
use std::fs::{self};
use std::{
    io,
    path::{Path, PathBuf},
};
use thiserror::Error;

use serde::{Deserialize, Serialize};

pub mod time {
    use chrono::{DateTime, NaiveDateTime, Utc};
    use serde::{Deserialize, Serialize};

    #[derive(Debug)]
    pub struct UnixTimestamp(pub DateTime<Utc>);
    const FORMAT: &'static str = "%s.%6f";

    impl Serialize for UnixTimestamp {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            let s = format!("{}", self.0.format(FORMAT));
            serializer.serialize_str(&s)
        }
    }

    impl<'de> Deserialize<'de> for UnixTimestamp {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let s = String::deserialize(deserializer)?;
            let dt = NaiveDateTime::parse_from_str(&s, FORMAT).map_err(serde::de::Error::custom)?;
            Ok(UnixTimestamp(DateTime::<Utc>::from_naive_utc_and_offset(
                dt, Utc,
            )))
        }
    }
}

#[derive(Debug, Error)]
pub enum WTSError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Repo dose not exist")]
    EmptyRepository,
    #[error("Object not found: {0}")]
    ObjectNotFound(String),
    #[error("Invalid reference: {0}")]
    InvalidReference(String),
    #[error("SafeTensor error: {0}")]
    SafeTensorError(String),
    #[error("Other error: {0}")]
    Other(String),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Commit {
    hash: Vec<u8>,
    parent_hash: Option<Vec<u8>>,
    timestamp: time::UnixTimestamp,
    message: String,
    metadata: serde_json::Value,
}

pub struct Refernce {
    pub name: String,
    pub commit_hash: [u8; 64],
}

pub struct OwnedSafeTensor {
    pub buffer: Vec<u8>,
    pub tensor: SafeTensors<'static>,
}

#[derive(Debug)]
pub struct Repository {
    pub root: PathBuf,
}

impl Repository {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let root = path.as_ref().to_path_buf();
        if !root.exists() {
            fs::create_dir_all(&root.join(".wts"))?;
        } 
        Ok(Self { root })
    }

    pub fn open() -> Result<Self, WTSError>{
        let root = std::env::current_dir().map_err(|e| WTSError::Io(e))?;
        if !root.exists() {
            return Err(WTSError::EmptyRepository);
        } 
        Ok(Self { root })
    }

    pub fn init(&self) -> io::Result<()> {
        let wts_dir = self.root.join(".wts");

        fs::create_dir_all(wts_dir.join("objects"))?;
        fs::create_dir_all(wts_dir.join("refs/heads"))?;
        fs::create_dir_all(wts_dir.join("refs/tags"))?;
        fs::create_dir_all(wts_dir.join("commits"))?;

        fs::write(wts_dir.join("HEAD"), r#"ref: ref/heads/main"#)?;

        Ok(())
    }

    pub fn is_initialized(&self) -> bool {
        self.root.join(".wts").exists()
    }

    pub fn create_commit(
        &self,
        tensors: &HashMap<String, Tensor>,
        message: &str,
        metadata: serde_json::Value,
        parent: Option<Vec<u8>>,
    ) -> Result<String, WTSError> {
        // Generate hash from tensors
        let hash = self.hash_tensors(tensors)?;

        let hex_hash = hex::encode(&hash);

        // Create commit object
        let commit = Commit {
            hash: hash.to_vec().clone(),
            parent_hash: parent,
            timestamp: UnixTimestamp(Utc::now()),
            message: message.to_string(),
            metadata,
        };

        // Save commit
        let commit_path = self
            .root
            .join(".wts")
            .join("commits")
            .join(format!("{}.json", hex_hash));

        let commit_json = serde_json::to_string_pretty(&commit)
            .map_err(|e| WTSError::SafeTensorError(e.to_string()))?;
        fs::write(commit_path, commit_json)?;

        // Save tensors
        self.store_tensors(tensors, &hex_hash)?;

        Ok(hex_hash)
    }

    pub fn hash_tensors(&self, tensors: &HashMap<String, Tensor>) -> Result<[u8; 64], WTSError> {
        let mut hasher = Sha512::new();

        // Sort keys to ensure deterministic hashing
        let mut keys: Vec<&String> = tensors.keys().collect();
        keys.sort();

        for key in keys {
            let tensor = tensors.get(key).unwrap();

            // Hash the tensor shape and dtype to ensure unique hashes for different structures
            hasher.update(key.as_bytes()); // Include layer name
            hasher.update(&tensor.to_string()); // Shape
            hasher.update(tensor.dtype().as_str().as_bytes()); // Data type

            // Hash the raw bytes of the tensor
            let mut buffer = Vec::new();
            tensor.write_bytes(&mut buffer).unwrap();
            hasher.update(&buffer);
        }

        let hash = hasher.finalize();

        // Ensure fixed 64-byte output
        let mut buffer = [0; 64];
        buffer.copy_from_slice(&hash);
        Ok(buffer)
    }

    fn store_tensors(&self, tensors: &HashMap<String, Tensor>, hash: &str) -> Result<(), WTSError> {
        let object_path = self.root.join(".wts").join("objects").join(hash);
        let _ = save(tensors, object_path)
            .map(|_| WTSError::SafeTensorError(format!("Could not save tensor")));
        Ok(())
    }

    pub fn create_branch(&self, name: &str, commit_hash: &str) -> Result<(), WTSError> {
        let ref_path = self.root.join(".wts").join("refs").join("heads").join(name);

        fs::write(ref_path, commit_hash)?;
        Ok(())
    }

    pub fn create_tag(&self, name: &str, commit_hash: &str) -> Result<(), WTSError> {
        let tag_path = self.root.join(".wts").join("refs").join("tags").join(name);

        fs::write(tag_path, commit_hash)?;
        Ok(())
    }

    pub fn get_commit(&self, hash: &str) -> Result<Commit, WTSError> {
        let commit_path = self
            .root
            .join(".wts")
            .join("commits")
            .join(format!("{}.json", hash));

        let commit_json = fs::read_to_string(commit_path)?;
        serde_json::from_str(&commit_json).map_err(|e| WTSError::SafeTensorError(e.to_string()))
    }

    pub fn get_reference(&self, ref_path: &str) -> Result<String, WTSError> {
        let full_path = self.root.join(".wts").join(ref_path);
        match fs::read_to_string(full_path) {
            Ok(hash) => Ok(hash.trim().to_string()),
            Err(_) => Err(WTSError::InvalidReference(ref_path.to_string())),
        }
    }

    pub fn get_obj(&self, path : &str, device : &Device) -> Result<HashMap<String, Tensor>, WTSError> {
        load(path, device).map_err(|e| WTSError::Other(e.to_string()))
    }
}

pub struct CommitIterator<'a> {
    repo: &'a Repository,
    current_hash: Option<Vec<u8>>,
}

impl<'a> Iterator for CommitIterator<'a> {
    type Item = Result<Commit, WTSError>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.current_hash {
            None => None,
            Some(hash) => {
                let hex_hash = hex::encode(&hash);
                let commit_result = self.repo.get_commit(&hex_hash);
                match &commit_result {
                    Ok(commit) => {
                        self.current_hash = commit.parent_hash.clone();
                        Some(commit_result)
                    }
                    Err(_) => {
                        self.current_hash = None;
                        Some(commit_result)
                    }
                }
            }
        }
    }
}
