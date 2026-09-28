/** @type {import('next').NextConfig} */
const isGithubActions = process.env.GITHUB_ACTIONS || false;
let repoName = '';

if (isGithubActions && process.env.GITHUB_REPOSITORY) {
  repoName = `/${process.env.GITHUB_REPOSITORY.split('/')[1]}`;
}

const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? (repoName || '/sigma-signal-analysis-SIH-2026');

const nextConfig = {
  output: 'export',
  basePath: basePath !== '/' ? basePath : '',
  images: {
    unoptimized: true,
  },
  reactStrictMode: true,
};

export default nextConfig;

