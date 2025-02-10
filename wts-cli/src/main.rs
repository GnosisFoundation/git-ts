use candle::safetensors::load;
use clap::{Parser, Subcommand};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};
use wts_core::{Repository, WTSError};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new WST repository
    Init {
        /// Optional path to initialize the repository
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Create a new commit
    Commit {
        /// Path to the SafeTensor file
        #[arg(short, long)]
        file: PathBuf,
        /// Commit message
        #[arg(short, long)]
        message: String,
        /// Optional metadata as JSON string
        #[arg(short = 'd', long)]
        metadata: Option<String>,
    },
    /// Create a new branch
    Branch {
        /// Name of the branch
        name: String,
        /// Commit hash to branch from
        #[arg(short, long)]
        commit: String,
    },
    /// Create a new tag
    Tag {
        /// Name of the tag
        name: String,
        /// Commit hash to tag
        #[arg(short, long)]
        commit: String,
    },
    /// Show commit information
    Show {
        /// Commit hash to show
        hash: String,
    },
    CatFile {
        hash: String,
    },
}

fn main() -> Result<(), WTSError> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Init { path } => {
            let repo = Repository::new(path)?;
            repo.init()?;
            println!("Initialized empty WST repository at {:?}", path);
        }
        Commands::Commit {
            file,
            message,
            metadata,
        } => {
            let repo = Repository::open()?;

            // Convert to absolute path
            let absolute_file_path = if file.is_absolute() {
                file.clone()
            } else {
                std::env::current_dir()?.join(file)
            };

            if !absolute_file_path.exists() {
                return Err(WTSError::Other(format!(
                    "SafeTensor file not found at: {:?}",
                    absolute_file_path
                )));
            }

            let data = std::fs::read(&absolute_file_path).map_err(|e| {
                println!("Error opening file: {:?}", e);
                WTSError::Io(e)
            })?;

            let tensors = load(absolute_file_path, &candle::Device::Cpu).map_err(|_| {
                WTSError::SafeTensorError(format!("Could not load tensor from file"))
            })?;

            let metadata_value = if let Some(meta_str) = metadata {
                serde_json::from_str(meta_str)
                    .map_err(|e| WTSError::SafeTensorError(e.to_string()))?
            } else {
                serde_json::Value::Null
            };

            let hash = repo.create_commit(&tensors, message, metadata_value, None)?;
            println!("Created commit: {}", hash);
        }
        Commands::Branch { name, commit } => {
            let repo = Repository::new(".")?;
            repo.create_branch(name, commit)?;
            println!("Created branch '{}' at {}", name, commit);
        }
        Commands::Tag { name, commit } => {
            let repo = Repository::new(".")?;
            repo.create_tag(name, commit)?;
            println!("Created tag '{}' at {}", name, commit);
        }
        Commands::Show { hash } => {
            let repo = Repository::new(".")?;
            let commit = repo.get_commit(hash)?;
            println!("{:#?}", commit);
        }
        Commands::CatFile { hash } => {
            let repo = Repository::open()?;

            let path = Path::new(".wts").join("objects").join(hash);
            let obj = repo.get_obj(path.to_str().unwrap(), &candle::Device::Cpu)?;

            println!("{obj:?}");
        }
    }

    Ok(())
}
