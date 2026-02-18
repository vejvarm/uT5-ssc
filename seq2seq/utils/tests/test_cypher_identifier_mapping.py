import unittest

from seq2seq.utils.cypher_identifier_mapping import (
    CypherIdentifierMappingBuilder,
    IdentifierMapping,
)


class CypherIdentifierMappingTests(unittest.TestCase):
    def test_strip_root_only_keeps_relationship_types_unprefixed(self):
        schema = {
            "NodeLabels": ["ROOT__course_arrange", "ROOT__teacher", "ROOT__course"],
            "RelationshipLabels": ["course_arrange__TEACHER_ID", "course_arrange__COURSE_ID"],
            "NodeProperties": [],
            "Relationships": [
                {
                    "relationshipType": "course_arrange__TEACHER_ID",
                    "startNodeLabels": ["ROOT__course_arrange"],
                    "endNodeLabels": ["ROOT__teacher"],
                },
                {
                    "relationshipType": "course_arrange__COURSE_ID",
                    "startNodeLabels": ["ROOT__course_arrange"],
                    "endNodeLabels": ["ROOT__course"],
                },
            ],
            "RelationshipProperties": {},
        }
        query = (
            "match (t3:ROOT__course_arrange)-[:course_arrange__TEACHER_ID]->(t1:ROOT__teacher) "
            "match (t3:ROOT__course_arrange)-[:course_arrange__COURSE_ID]->(t2:ROOT__course) "
            "where t1:ROOT__teacher "
            "return t1.teacher__name as t1_name, t2.course__course as t2_course"
        )

        mapping = CypherIdentifierMappingBuilder().build(schema, strategy="strip_root_only")
        shortened = mapping.shorten_query(query)
        restored = mapping.restore_query(shortened)

        self.assertEqual(query, restored)
        self.assertIn("[:course_arrange__TEACHER_ID]", restored)
        self.assertIn("[:course_arrange__COURSE_ID]", restored)
        self.assertEqual({}, mapping.forward_by_context.get("relationship_type", {}))
        self.assertNotEqual({}, mapping.forward_by_context.get("node_label", {}))

    def test_strip_prefix_round_trip_restores_relationship_types(self):
        schema = {
            "NodeLabels": ["ROOT__course_arrange", "ROOT__teacher"],
            "RelationshipLabels": ["ROOT__course_arrange__TEACHER_ID"],
            "NodeProperties": [],
            "Relationships": [
                {
                    "relationshipType": "ROOT__course_arrange__TEACHER_ID",
                    "startNodeLabels": ["ROOT__course_arrange"],
                    "endNodeLabels": ["ROOT__teacher"],
                }
            ],
            "RelationshipProperties": {},
        }
        query = "match (a:ROOT__course_arrange)-[:ROOT__course_arrange__TEACHER_ID]->(b:ROOT__teacher) return a"

        mapping = CypherIdentifierMappingBuilder().build(schema, strategy="strip_prefix")
        shortened = mapping.shorten_query(query)
        restored = mapping.restore_query(shortened)

        self.assertEqual(query, restored)
        self.assertIn("[:TEACHER_ID]", shortened)
        self.assertNotEqual({}, mapping.forward_by_context.get("relationship_type", {}))

    def test_legacy_label_context_payload_still_restores(self):
        payload = {
            "contexts": {
                "label": {
                    "original": ["ROOT__teacher"],
                    "short": ["teacher"],
                }
            }
        }
        mapping = IdentifierMapping.from_serializable(payload)
        restored = mapping.restore_query("match (t:teacher) return t")

        self.assertEqual("match (t:ROOT__teacher) return t", restored)


if __name__ == "__main__":
    unittest.main()
