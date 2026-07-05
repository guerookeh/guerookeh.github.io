import { FullSlug, resolveRelative } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const AllTags: QuartzComponent = ({ allFiles, fileData }: QuartzComponentProps) => {
  const counts = new Map<string, number>()

  for (const file of allFiles) {
    for (const tag of file.frontmatter?.tags ?? []) {
      counts.set(tag, (counts.get(tag) ?? 0) + 1)
    }
  }

  const tags = [...counts.entries()].sort(([a], [b]) => a.localeCompare(b))

  return (
    <nav class="all-tags" aria-label="Tags">
      <h3>Tags</h3>
      <ul>
        {tags.map(([tag, count]) => (
          <li>
            <a class="internal tag-link" href={resolveRelative(fileData.slug!, `tags/${tag}` as FullSlug)}>
              {tag}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  )
}

AllTags.css = `
.all-tags > h3 {
  margin: 0 0 0.75rem;
  font-size: 1rem;
}

.all-tags > ul {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem 0.35rem;
  list-style: none;
  margin: 0;
  padding: 0;
}

.all-tags > ul > li {
  display: inline-flex;
  margin: 0;
  font-size: 0.88rem;
}

.all-tags > ul > li > a.tag-link {
  border-radius: 7px;
  font-size: 0.88rem;
  line-height: 1.2;
  padding: 0.16rem 0.36rem;
}
`

export default (() => AllTags) satisfies QuartzComponentConstructor
